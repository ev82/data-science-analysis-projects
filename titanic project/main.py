# %%
# all imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score 

# %% 
# get data
training = pd.read_csv('titanic data/train.csv')
test = pd.read_csv('titanic data/test.csv')

# quick look at the data
# %%
training.head()
# %%
training.describe()
# %%
#quick way to separate numeric columns
training.describe().columns
# %%
training.describe(include=['O'])

# %%
# look at numeric and categorical values separately 
df_num = training[['Age','SibSp','Parch','Fare']]
df_cat = training[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]

# %% 
# distributions for all numeric variables 
for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()

# %%
# compare survival rate across Age, SibSp, Parch, and Fare 
pd.pivot_table(training, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])


# comparing survival and each of these categorical variables 
# %%
pd.pivot_table(training, index = 'Survived', columns = 'Pclass', values = 'Ticket' ,aggfunc ='count')
# %%
pd.pivot_table(training, index = 'Survived', columns = 'Sex', values = 'Ticket' ,aggfunc ='count')
# %%
pd.pivot_table(training, index = 'Survived', columns = 'Embarked', values = 'Ticket' ,aggfunc ='count')

# %%
# simplify cabins
training.Cabin
training['cabin_multiple'] = training.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
# after looking at this, we may want to look at cabin by letter or by number. Let's create some categories for this 
# letters 
# multiple letters 
training['cabin_multiple'].value_counts()

# %%
pd.pivot_table(training, index = 'Survived', columns = 'cabin_multiple', values = 'Ticket' ,aggfunc ='count')

# %%
# creates categories based on the cabin letter (n stands for null)
# in this case we will treat null values like it's own category
training['cabin_adv'] = training.Cabin.apply(lambda x: str(x)[0])

# %%
# comparing surivial rate by cabin
print(training.cabin_adv.value_counts())
print()
pd.pivot_table(training,index='Survived',columns='cabin_adv', values = 'Name', aggfunc='count')

# %%
# understand ticket values better 
# numeric vs non numeric 
training['numeric_ticket'] = training.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
training['ticket_letters'] = training.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)

# %%
training['numeric_ticket'].value_counts()

# %%
# difference in numeric vs non-numeric tickets in survival rate 
pd.pivot_table(training,index='Survived',columns='numeric_ticket', values = 'Ticket', aggfunc='count')

# %%
# survival rate across different ticket types 
pd.pivot_table(training,index='Survived',columns='ticket_letters', values = 'Ticket', aggfunc='count')

# %%
# feature engineering on person's title 
training.Name.head(50)
training['name_title'] = training.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
#mr., ms., master. etc

# %%
training['name_title'].value_counts()

# data processing for model
# %%
# create all categorical variables that we did above for both training and test sets seperately
for df in [training, test]:
    df['cabin_multiple'] = df.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
    df['cabin_adv'] = df.Cabin.apply(lambda x: str(x)[0])
    df['numeric_ticket'] = df.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
    df['ticket_letters'] = df.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.', '').replace('/', '').lower() if len(x.split(' ')[:-1]) > 0 else 0)
    df['name_title'] = df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

# impute, ALWAYS derive stats from training only, then apply to both
age_median = training.Age.median()
fare_median = training.Fare.median()

training.Age = training.Age.fillna(age_median)
test.Age = test.Age.fillna(age_median)
training.Fare = training.Fare.fillna(fare_median)
test.Fare = test.Fare.fillna(fare_median)

# drop missing Embarked values
training.dropna(subset=['Embarked'], inplace=True)
test.dropna(subset=['Embarked'], inplace=True)  # likely a no-op, just in case

# log norm of fare
training['norm_fare'] = np.log(training.Fare + 1)
test['norm_fare'] = np.log(test.Fare + 1)

# Pclass to string for both sets seperately
training.Pclass = training.Pclass.astype(str)
test.Pclass = test.Pclass.astype(str)

# encoding: FIT on training only, but TRANSFORM both

cat_cols = ['Pclass', 'Sex', 'Embarked', 'cabin_adv', 'name_title']
num_cols = ['Age', 'SibSp', 'Parch', 'norm_fare', 'cabin_multiple', 'numeric_ticket']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(training[cat_cols])   # only looks at training categories

X_train = pd.concat([
    training[num_cols].reset_index(drop=True),
    pd.DataFrame(encoder.transform(training[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))
], axis=1)

X_test = pd.concat([
    test[num_cols].reset_index(drop=True),
    pd.DataFrame(encoder.transform(test[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))
], axis=1)

y_train = training.Survived

# scale data, fit on training set only, apply to both sets
# %%
scale = StandardScaler()
scale_cols = ['Age', 'SibSp', 'Parch', 'norm_fare']

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[scale_cols] = scale.fit_transform(X_train[scale_cols])
X_test_scaled[scale_cols] = scale.transform(X_test[scale_cols])

# model building, use logisitic regression, k nearest neighbor, and XGB (xtreme gradient boosting)
# %%
# start with logistic regression
lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X_train,y_train,cv=5)
print(cv)
print("Logistic Regression: " + str(cv.mean()))

# %%
# logisitic regression with scaled training to see if scaling matters
lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X_train_scaled,y_train,cv=5)
print(cv)
print("Logistic Regression (scaled training): " + str(cv.mean()))

# %%
# k nearest neighbor
knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train,y_train,cv=5)
print(cv)
print("KNeighbors: " + str(cv.mean()))

# %%
# k nearest neighbor but scaled training
knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train_scaled,y_train,cv=5)
print(cv)
print("KNeighbors (scaled training): " + str(cv.mean()))

# %%
# XGB model, just scaled training 
xgb = XGBClassifier(random_state =1)
cv = cross_val_score(xgb,X_train_scaled,y_train,cv=5)
print(cv)
print("XGB (scaled): " + str(cv.mean()))


# most of the code is from Ken Jee's walkthrough