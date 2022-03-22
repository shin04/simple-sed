#!/bin/bash

#----- load .env file -----#
source ./.env

#----- synbolic link -----#
cd $WORK_DIR
mkdir log
ln -s $META_PATH ./meta
ln -s $MODEL_PATH ./models
ln -s $RESULT_PATH ./results
mkdir ./dataset
cd ./dataset
ln -s $AUDIO_PATH ./audio
ln -s $FEAT_PATH ./feat

#----- generate .env.mlflow -----#
cd $WORK_DIR
touch .env.mlflow
echo "TRACKING_URL=$MLFLOW_TRACKING_PATH" > .env.mlflow

#----- venv -----#
# downstream task
cd $WORK_DIR
python3 -m venv .venv
source .venv/bin/activate
pip install -r ./requirements.txt
pip install install \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111 \
    torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html
deactivate

# finetuning task
python3 -m venv .venv-finetune
source .venv-finetune/bin/activate
pip install -r ./requirements.finetune.txt
pip install install \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111 \
    torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html
cd $FAIRSEQ_DIR
pip install --editable ./
deactivate

cd $WORK_DIR
