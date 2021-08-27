# tau3mu-bds

## Requirements:

numpy == 1.20.2
torch == 1.8.1
scikit-learn==0.24.1
matplotlib==3.4.1

## Singularity Image
Link to CERNBOX:
https://cernbox.cern.ch/index.php/s/xUKjWDfNXqeq3A4

Place gnn_siqi.img into Working

## Training a Model
cd Working

singularity run gnn_siqi.img

python retrain_model.py


