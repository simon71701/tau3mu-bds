# tau3mu-bds

## Requirements:

numpy == 1.19.5

torch == 1.8.1

scikit-learn==0.24.2

matplotlib==3.4.2

pandas == 1.2.4

## Making a Dataset

Edit genNpArrays.py
- Edit the variable interested_vars to change what characteristics to use in the dataset
- Use the transform and naive keyword arguments to determine if a transformation will be applied to mu_hit_sim_phi
- - transform=True, naive=False: Principle direction transformation. (Needs to be checked, do not use for the time being)
- - transform=True, naive=True: mu_hit_sim_phi replaced with the two variables mu_hit_cos_phi and mu_hit_sin_phi
- Use the comment keyword argument to add additional specifications to the default naming

## Training a Model
Run the following commands:

cd Working
python main.py

Be sure to load the correct datasets.

