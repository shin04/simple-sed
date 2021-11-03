$WORK_DIR=$PWD

#----- synbolic link -----#
ln -s $HOME/mrnas02home/datasets/urban_sed/meta ./meta
ln -s $HOME/nas02home/models/sed ./models
ln -s $HOME/nas02home/results/sed ./results
mkdir ./dataset
cd ./dataset
ln -s $HOME/mrnas02home/datasets/urban_sed/audio ./audio
ln -s $HOMEnas02home/dataset/hubert_feat/urbansed_audioset ./feat


#----- venv -----#
cd $WORK_DIR
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install install \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111 \
    torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html