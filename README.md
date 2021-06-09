# tau3mu-bds

## Requirements:

numpy == 1.19.2

matplotlib == 3.4.1

torch == 1.8.1

pandas == 1.2.4

tqdm == 4.60.0

sklearn == 0.0

## Training a Model

Default arguments:

batch_size=50
dropout=.1
epochs=250
lr=.0001
pileup=0
num_funnel_layers=5
maxhits=72
extra_filter=0
mix=0
early_stop=0

Run the following:

cd Working

python funnel1_4.py --args

## Testing a Model

Default arguments:

batch_size=50
dropout=.1
epochs=250
lr=.0001
pileup=0
num_funnel_layers=5
maxhits=72
extra_filter=0
mix=0
test_on=200
early_stop=0

Run the following:

cd Working

python test_model1_4.py --args
