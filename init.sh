#----- synbolic link -----#
ln -s $HOME/mrnas02home/datasets/urban_sed/meta ./meta
ln -s $HOME/nas02home/models/sed ./models
ln -s $HOME/nas02home/results/sed ./results
mkdir ./dataset
cd ./dataset
ln -s $HOME/mrnas02home/datasets/urban_sed/audio ./audio
ln -s $HOMEnas02home/dataset/hubert_feat/urbansed_audioset ./feat
