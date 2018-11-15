# ANN-Pipeline
Scripts for artificial neural networks for the Shiu Lab

## Environment Requirements
* python                    3.6
* numpy                     1.11.3
* pandas                    0.18.1
* tensorflow                

Tensorflow typically requires special installation. See the [HPCC instructions](https://wiki.hpcc.msu.edu/display/ITH/TensorFlow) at MSU as an example. 
    
MSU HPCC: source /mnt/home/azodichr/python3-tfcpu/bin/activate
Calculon2: TBD


## Multi-Layer Perceptrons
Use this code to generate fully connected artificial neural networks. Note that a few features (e.g.: saving trained network weights) are only available for networks with =< 3 hidden layers. 


To Run:

```python ANN_mlp.py -f full -x [path/to/feature_data] -y [path/to/y_data] -y_name Y_Col_Name -ho [path/to/holdout_list] -save OUTPUT_NAME```
* For more info/additional options run ANN_mlp.py with no parameters

Real Example:
Break up the run by running 1 replicate of the grid search 10 times (-f gs -gs_reps 1), once done, use grid search results as input to build final models (-f run -params SAVE_GridSearch.txt)

```
for i in $(seq 1 10); do python ANN_mlp.py -f gs -x geno.csv -y pheno.csv -y_name HT -sep ',' -ho holdout.txt -save mlp_HT -gs t -gs_reps 1 -weights xavier -norm t; done
python ANN_mlp.py -f run -x geno.csv -y pheno.csv -y_name HT -sep ',' -ho holdout.txt -save mlp_HT_final -params mlp_HT_GridSearch.txt -weights xavier -norm t
```
* For more info/additional options run ANN_mlp.py with no parameters



## Data Preprocessing
Some data preprocessing may be required before model building. These scripts are available from the [ML-Pipeline](https://github.com/ShiuLab/ML-Pipeline) repository. See README.md in that repository for more updated instructions.
### Define Holdout Set (Azodi)
Randomly select X% of instances to be held out for feature selection and model training (imputation will happen on all data). For classification models, holds out X% from each class

Example:

```python ML_Pipeline/holdout.py -df [path/to/dataframe] -type c -p 0.1```
* For more info/additional options run Feature_Selection.py with no parameters
    
### Feature Selection (Azodi)
Available feature selection tools: RandomForest, Chi2, LASSO (L1 penalty), enrichement (Fisher's Exact test - binary features only), and BayesA (regression only). For parameters needed for each feature selection algorithm run Feature_Selection.py with no parameters.

Example:
```
export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
python ML_Pipeline/Feature_Selection.py -df [path/to/dataframe] -f [rf/chi2/lasso/fet/bayesA] -n [needed for chi2/rf/bayesA] -p [needed for LASSO/FET] -type [needed for LASSO] -ho [needed if using holdout set]
```
* For more info/additional options run holdout.py with no parameters

