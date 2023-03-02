#%% #* Importing all the necessary libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from numpy import absolute
# %% #* Read the csv file for training
train_df = pd.read_csv('train.csv')
train_df

X_test = pd.read_csv('test.csv')
X_test

#Split the train dataframe for validation and training
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.3, 
                                                random_state=1121218)
                                                

# %% #* Checks on datatypes null values and unique counts

def check_columns(df):
    
    column = []
    shape = []
    datatype = []
    unique_values = []
    null_values = []
    nulls_count =[]
    nunique = []

    col_check = pd.DataFrame()
    
    for col_name in df.columns:
        column.append(col_name)
        shape.append(df[col_name].shape)
        datatype.append(df[col_name].dtype)
        unique_values.append(df[col_name].is_unique)
        null_values.append(df[col_name].isnull().any())
        nulls_count.append(df[col_name].isna().sum())
        nunique.append(df[col_name].nunique())
      
    
    col_check['column'] = column
    col_check['shape'] = shape
    col_check['datatype'] = datatype
    col_check['unique_values'] = unique_values
    col_check['null_values'] = null_values
    col_check['null_count'] = nulls_count
    col_check['nunique'] = nunique
    
    return col_check 

check_data = check_columns(X_train).T
check_data


# %% #! EDA (Not necessary, only for graphs)

#Describe numerical columns
X_train.describe().iloc[:,:10]

#Describe categorical columns
X_train.describe(include='object').T

#No duplicates in the dataframe
X_train[X_train.duplicated(keep=False)]

#Check the dtypes of categorical and numerical columns
X_train.dtypes
numerical_features = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
len(numerical_features)

categorical_features = [col for col in X_train.columns if X_train[col].dtype in ['object']]
len(categorical_features)

# ******************* OR SIMPLY use sklearn pipeline feature to get the numerical and cate columns*********

numerical_features = X_train.select_dtypes(include='number').columns.tolist()
numerical_features
categorical_features = X_train.select_dtypes(exclude='number').columns.tolist()

# %% #! check null values in categorical and numerical columns (Not necessary, only for graphs)

#For categorical variables
#Removing few categorical columns in the data as they have many missing values
#remove_list = ['Alley','PoolQC','Fence','MiscFeature','FireplaceQu']
categorical_features.remove('Alley')
categorical_features.remove('PoolQC')
categorical_features.remove('Fence')
categorical_features.remove('MiscFeature')
categorical_features.remove('FireplaceQu')

X_train[categorical_features].describe().iloc[:, 16:]
#Replacing the nan values with mode in categorical columns

cols = ['MasVnrType','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','Electrical',
        'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

for i in cols:
    print(i)
    X_train[i] = X_train[i].fillna(X_train[i].mode()[0])
    X_train[i].isnull().any()

check_columns(X_train[categorical_features])


#For numerical variables

cols = ['LotFrontage','MasVnrArea','GarageYrBlt']

for i in cols:
    print(i)
    X_train[i] = X_train[i].fillna(X_train[i].mean())
    X_train[i].isnull().any()

check_columns(X_train[numerical_features])

# %% #! Plotting graphs (Not necessary, only for graphs)

#Scatter plots for categorical variables
fig, axs = plt.subplots(nrows =5, ncols = 6, figsize = (30,15))

for i, col_name in enumerate(X_train[categorical_features].columns):
    row = i//27
    col = i%27
    axs = axs.T.flatten()
    axs[col].scatter(X_train[col_name], y_train, alpha = 0.4)
    axs[col].set_xlabel(col_name)
    axs[col].set_ylabel('Sales price')
    axs[col].tick_params(axis='x', labelrotation=90)
plt.show()

#Dist plots for numerical variables
plt.figure(figsize = (25, 25))
for i in enumerate(X_train[numerical_features].columns):
    #print("the i:", i)
    #print("the i0:", i[0])
    #print("the i1:", i[1])
    plt.subplot(6, 7,i[0]+1)
    #sns.countplot(i[1],data = df)
    sns.distplot(X_train, x = X_train[i[1]])
    plt.title(i[1])

#Box plots for numerical variables
sns.boxplot(x="variable", y="value", data=pd.melt(X_train[numerical_features].iloc[:,:10]))
plt.xticks(rotation=90).figure(figsize = (15, 15))

sns.boxplot(x="variable", y="value", data=pd.melt(X_train[numerical_features].iloc[:,10:21]))
plt.xticks(rotation=90).figure(figsize = (15, 15))

sns.boxplot(x="variable", y="value", data=pd.melt(X_train[numerical_features].iloc[:,21:31]))
plt.xticks(rotation=90).figure(figsize = (15, 15))

sns.boxplot(x="variable", y="value", data=pd.melt(X_train[numerical_features].iloc[:,31:]))
plt.xticks(rotation=90).figure(figsize = (15, 15))

# %% #* Plot correlation graph for numerical variables

corr = X_train[numerical_features].corr()
plt.figure(figsize=(35,25))
sns.heatmap(corr, annot=True, cmap='coolwarm')

# %% #* Creating pipelines 

numerical_features = X_train.select_dtypes(include='number').columns

#Removing few categorical columns in the data as they have many missing values
#X_train= X_train.drop(['Alley','PoolQC','Fence','MiscFeature','FireplaceQu'], axis=1)

categorical_features = X_train.select_dtypes(exclude='number').columns

numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

#Fit and transform using column transformer
full_processor = ColumnTransformer(transformers=[
    ('number', numeric_pipeline, numerical_features),
    ('category', categorical_pipeline, categorical_features)
])

#%% #*Fit and transform using column transformer

#Define model
lasso = Lasso(alpha=0.1)

#Define pipeline for model
lasso_pipeline = Pipeline(steps=[('preprocess', full_processor),
                                 ('model', lasso)
                                 ])

#Main hyperparameter for Lasso is alpha which can range from 0 to infinity. 
#For simplicity, we will only cross-validate on the values within 0 and 1 with steps of 0.05

param_dict = {'model__alpha': np.arange(0, 1, 0.05)}

search = GridSearchCV(lasso_pipeline, param_dict, 
                      cv=10, 
                      scoring='neg_mean_absolute_error')

_ = search.fit(X_train, y_train)

#* Print the best scores
print('Best score:', abs(search.best_score_))
print('Best alpha:', search.best_params_)

# %% #*Training with different alpha value

#As you can see, best alpha is 0.95 which is the very end of our given interval, i. e. [0, 1) with a step of 0.05.
#We need to search again in case the best parameter lies in a bigger interval:

param_dict = {'model__alpha': np.arange(1, 100, 5)}

search_again = GridSearchCV(lasso_pipeline, param_dict, 
                      cv=10, 
                      scoring='neg_mean_absolute_error')

_ = search_again.fit(X_train, y_train)

print('Best score:', abs(search_again.best_score_))
print('Best alpha:', search_again.best_params_)

# %% #* Rebuilding with new alpha values

#With the best hyperparameters, we get a significant drop in MAE (which is good).
#Let’s redefine our pipeline with Lasso(alpha=76):

lasso = Lasso(alpha=76)

final_lasso_model = Pipeline(steps=[('preprocess', full_processor),
                                    ('model',lasso)])


_ = final_lasso_model.fit(X_train, y_train)
preds = final_lasso_model.predict(X_valid)
mean_absolute_error(y_valid,preds)
#r2_score(y_valid,preds)

#%% #* Predictions with the test variables

pred_test = final_lasso_model.predict(X_test)

output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': pred_test})

output

# %% #* Predicting with XG Boost model

xgb = XGBRegressor(learning_rate = 0.05)

xgb_pipeline = Pipeline(steps=[('preprocess', full_processor),('model', xgb)])

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#crossvalidation and scoring
scores = cross_val_score(xgb_pipeline, X_train, y_train,
                              cv=cv,
                              scoring="neg_mean_absolute_error")

print("MAE score:\n", scores.mean())
absolute(scores)

#%% #* Hyperoptimization
param_grid = {
    "model__learning_rate": np.arange(0.01,0.3,0.08),
    "model__max_depth":np.arange(1,10,1)
}

hyper = GridSearchCV(
    estimator = xgb_pipeline,
    param_grid = param_grid ,
    scoring = "neg_mean_absolute_error",
    verbose = 10,
    cv = cv)

# Fit
hyper.fit(X_train,y_train)

# %% #* Predictions with XBG
print(hyper.best_score_)
print(hyper.best_estimator_)

xgb_predict = hyper.best_estimator_.predict(X_valid)
test_score = r2_score(y_valid,xgb_predict)
test_score
# Got the R2 score of 89.34 with XGB classifier which outperforms the lasso model which has only R2 score of 77.39 

pred_test = hyper.best_estimator_.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': pred_test})

output
# %%
