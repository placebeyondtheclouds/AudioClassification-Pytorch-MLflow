import os
import random

import numpy as np
from torch.utils.data import Dataset

from macls.data_utils.audio import AudioSegment
from macls.utils.logger import setup_logger

import io # by placebeyondtheclouds
import lmdb # by placebeyondtheclouds

logger = setup_logger(__name__)


class CustomDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 do_vad=True,
                 max_duration=3,
                 min_duration=0.5,
                 mode='train',
                 sample_rate=16000,
                 aug_conf={},
                 num_speakers=1000,
                 use_dB_normalization=True,
                 target_dB=-20,
                 lmdb_path=None): # by placebeyondtheclouds
        """音频数据加载器

        Args:
            data_list_path: 包含音频路径和标签的数据列表文件的路径
            do_vad: 是否对音频进行语音活动检测（VAD）来裁剪静音部分
            max_duration: 最长的音频长度，大于这个长度会裁剪掉
            min_duration: 过滤最短的音频长度
            aug_conf: 用于指定音频增强的配置
            mode: 数据集模式。在训练模式下，数据集可能会进行一些数据增强的预处理
            sample_rate: 采样率
            num_speakers: 总说话人数量
            use_dB_normalization: 是否对音频进行音量归一化
            target_dB: 音量归一化的大小
        """
        super(CustomDataset, self).__init__()
        self.do_vad = do_vad
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.mode = mode
        self._target_sample_rate = sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        self.aug_conf = aug_conf
        self.num_speakers = num_speakers
        self.noises_path = None
        # 获取数据列表
        with open(data_list_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()


        # by placebeyondtheclouds
        self.lmdb_env = None
        self.lmdb_path = lmdb_path
    def _init_lmdb(self): 
        self.lmdb_env = lmdb.open(self.lmdb_path, subdir=os.path.isdir(self.lmdb_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False, create=False)
        self.lmdb_txn = self.lmdb_env.begin(write=False, buffers=True)
        # with self.lmdbenv.begin(write=False) as lmdb_txn:
        #     self.lmdb_keys = pa.deserialize(lmdb_txn.get(b'__keys__'))

    def _read_lmdb(self, key):
        lmdb_data = self.lmdb_txn.get(key.encode())
        # lmdb_data = io.BytesIO(lmdb_data)
        return lmdb_data
        # by placebeyondtheclouds


    def __getitem__(self, idx):
        # 分割音频路径和标签
        audio_path, label = self.lines[idx].strip().split('\t')

        # by placebeyondtheclouds
        if self.lmdb_env is None and self.lmdb_path:
            self._init_lmdb()
        if self.lmdb_path and (self.mode == 'train' or self.mode == 'eval'):
            audio_segment = AudioSegment.from_bytes(self._read_lmdb(audio_path))
        else:
            # 读取音频 # by placebeyondtheclouds
            audio_segment = AudioSegment.from_file(audio_path) # by placebeyondtheclouds
        # by placebeyondtheclouds

        # 裁剪静音
        if self.do_vad:
            audio_segment.vad()
        # 数据太短不利于训练
        if self.mode == 'train' or self.mode == 'val': # ---- drop short audios during validation also---- by placebeyondtheclouds 
            if audio_segment.duration < self.min_duration:
                return self.__getitem__(idx + 1 if idx < len(self.lines) - 1 else 0)
        # 重采样
        if audio_segment.sample_rate != self._target_sample_rate:
            audio_segment.resample(self._target_sample_rate)
        # 音频增强
        if self.mode == 'train':
            audio_segment = self.augment_audio(audio_segment, **self.aug_conf)
        # decibel normalization
        if self._use_dB_normalization:
            audio_segment.normalize(target_db=self._target_dB)
        # 裁剪需要的数据
        audio_segment.crop(duration=self.max_duration, mode=self.mode)
        return np.array(audio_segment.samples, dtype=np.float32), np.array(int(label), dtype=np.int64)

    def __len__(self):
        return len(self.lines)

    # 音频增强
    def augment_audio(self,
                      audio_segment,
                      speed_perturb=False,
                      volume_perturb=False,
                      volume_aug_prob=0.2,
                      noise_dir=None,
                      noise_aug_prob=0.2):
        # 语速增强，注意使用语速增强分类数量会大三倍
        if speed_perturb:
            speeds = [1.0, 0.9, 1.1]
            speed_idx = random.randint(0, 2)
            speed_rate = speeds[speed_idx]
            if speed_rate != 1.0:
                audio_segment.change_speed(speed_rate)
        # 音量增强
        if volume_perturb and random.random() < volume_aug_prob:
            min_gain_dBFS, max_gain_dBFS = -15, 15
            gain = random.uniform(min_gain_dBFS, max_gain_dBFS)
            audio_segment.gain_db(gain)
        # 获取噪声文件
        if self.noises_path is None and noise_dir is not None:
            self.noises_path = []
            if noise_dir is not None and os.path.exists(noise_dir):
                for file in os.listdir(noise_dir):
                    self.noises_path.append(os.path.join(noise_dir, file))
        # 噪声增强
        if len(self.noises_path) > 0 and random.random() < noise_aug_prob:
            min_snr_dB, max_snr_dB = 10, 50
            # 随机选择一个noises_path中的一个
            noise_path = random.sample(self.noises_path, 1)[0]
            # 读取噪声音频
            noise_segment = AudioSegment.slice_from_file(noise_path)
            # 如果噪声采样率不等于audio_segment的采样率，则重采样
            if noise_segment.sample_rate != audio_segment.sample_rate:
                noise_segment.resample(audio_segment.sample_rate)
            # 随机生成snr_dB的值
            snr_dB = random.uniform(min_snr_dB, max_snr_dB)
            # 如果噪声的长度小于audio_segment的长度，则将噪声的前面的部分填充噪声末尾补长
            if noise_segment.duration < audio_segment.duration:
                diff_duration = audio_segment.num_samples - noise_segment.num_samples
                noise_segment._samples = np.pad(noise_segment.samples, (0, diff_duration), 'wrap')
            # 将噪声添加到audio_segment中，并将snr_dB调整到最小值和最大值之间
            audio_segment.add_noise(noise_segment, snr_dB)
        return audio_segment
