This is a data science/analysis project using the housing prices dataset from Kaggle.
The code is all python and all imports need to be installed for main.py to run. 
Opening the folder in an IDE with all the installed imports (sklearn, pandas, etc) should allow the code to run as is.

First section of the program are imports (packages like pandas, sklearn, etc).
Next is looking at the data to get a general understanding of what the dataset is like, including checking for missing values and looking at the distribution of the target variable (SalePrice)
After that I built a preprocessing pipeline that fills in missing values, scales numeric features, and one-hot encodes categorical features, fitting only on the training set.
Then I built and compared 4 models (Linear Regression, Random Forest, XGBoost, and a neural network) using grid search and cross-validation.
Next I used PCA to reduce the features into fewer dimensions and reran the same 4 models to see if it improved performance.After that I engineered new features from existing columns (like property age and total square footage) and reran the models again with this new feature set.
After that I engineered new features from existing columns (like property age and total square footage) and reran the models again with this new feature set.
Finally, I combined all the tuned models into a stacking ensemble, tested a few different meta-models, and used the best-performing models to generate the submission files.