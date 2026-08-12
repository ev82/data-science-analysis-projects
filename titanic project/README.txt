This is a data science/analysis project using the Titanic dataset from Kaggle.
The code is all python and all imports need to be installed for main.py to run. 
Opening the folder in an IDE with all the installed imports (sklearn, pandas, etc) should allow the code to run as is.

First section of the program are imports (packages like pandas and the dataset).
Next is looking at the data to get a general understanding of what the dataset is like.
After that I engineered new features into the dataset from existing columns (simplifying cabins, etc) and saw how each one related to survival.
Then I applied that feature engineering to both the training and test datasets, and cleaned the data by filling in missing values (this dataset didn't need any other cleaning).
I fit the encoding and scaling steps on the training set only, so the test set isn't used until prediction.
Then I built and compared 3 models, Logistic Regression, K-Nearest Neighbors, and XGBoost, tested both scaled and unscaled versions of the data.