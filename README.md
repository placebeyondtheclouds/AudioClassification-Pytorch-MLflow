# work in progress

old instructions do not apply anymore

metrics are modified for binary classification. multiclass classification needs further modifications

### need to install:

- mlflow
  - `conda install mlflow`
- LMDB
  - `conda install conda-forge::python-lmdb`

### need to start before training:

- `mlflow server --host CHANGEME --port 5008`
  - dashboard: http://CHANGEME:5008
  - make sure it's running:
    - `netstat -ln | grep 5008`
    - `ps -aux | grep mlflow`
  - kill it if necessary:
    - `ps -ef | grep 'mlflow.server' | grep -v grep | awk '{print $2}' | xargs -r kill -15`

### training

- `CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py`

### changes made to the code:

- in train.py added reading local rank from environment variable

- in trainer.py

  - added using LMDB as a data source
  - added MLflow logging
  - checkpoints of all epochs are now saved (not just the last 3)

- in predict.py

  - edited predict and predict_batch functions to ALSO return the `result` variable (**breaking changes**), needed for my custom inference code

- in cam++.yml

  - moved dataset folder outside the folder code
  - added MLflow parameters
  - added new parameters for changes in predict.py

- in reader.py

  - added LMDB option

- in .gitignore

  - added the default exclusions

### to do:

- also add jupyter notebook with
  - LMDB creation code
  - file list creation code
  - evaluation code
  - fix metrics for multilabel classification
