WORK_DIR=$PWD

#----- synbolic link -----#
ln -s $HOME/mrnas02home/datasets/urban_sed/meta ./meta
ln -s $HOME/nas02home/models/sed ./models
ln -s $HOME/nas02home/results/sed ./results
mkdir ./dataset
cd ./dataset
ln -s $HOME/mrnas02home/datasets/urban_sed/audio ./audio
ln -s $HOME/nas02home/dataset/hubert_feat/urbansed_audioset ./feat

#----- generate .env.mlflow -----#
cd $WORK_DIR
touch .env.mlflow
echo "TRACKING_URL=$WORK_DIR/results/mlflow" > .env.mlflow

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
cd ../fairseq
pip install --editable ./
deactivate

cd ../simple-sed