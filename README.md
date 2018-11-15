# ANN-Pipeline
Scripts for artificial neural networks for the Shiu Lab

## Environment Requirements
* biopython                 1.68
* matplotlib                1.5.1
* numpy                     1.11.3
* pandas                    0.18.1
* python                    3.4.4
* scikit-learn              0.18.1
* scipy                     0.18.1

Example: 

    wget http://repo.continuum.io/miniconda/Miniconda3-3.7.0-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda install biopython
    conda install matplotlib
    conda install pandas
    conda install scikit-learn
    
MSU HPCC: export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH

Calculon2: export PATH=/home/azodi/miniconda/bin:$PATH


## Multi-Layer Perceptrons
Use this code to generate fully connected artificial neural networks. Note that a few features (e.g.: saving trained network weights) are only available when there are 3 or fewer hidden layers. 





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

