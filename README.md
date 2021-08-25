# tau3mu-bds

## Requirements:

numpy == 1.19.2

matplotlib == 3.4.1

torch == 1.8.1

pandas == 1.2.4

tqdm == 4.60.0

sklearn == 0.0

## Singularity Image
Link to CERNBOX

Place gnn_siqi.img into Working

## Training a Model
cd Working

singularity run gnn_siqi.img

python retrain_model.py


