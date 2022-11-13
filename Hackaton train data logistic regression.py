### Pandas and Numpy
import pandas as pd
import numpy as np
### Visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt
### For Q-Q Plot
import scipy.stats as stats
### To ignore warnings
import warnings
warnings.filterwarnings('ignore')
### Machine Learning libraries
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import  LogisticRegression
from sklearn.model_selection import train_test_split
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest,chi2
### To be able to see maximum columns on screen
pd.set_option('display.max_columns', 500)
### To save the model
import pickle


# Data ingestion
df = pd.read_csv(r"https://raw.githubusercontent.com/Sahiljosan/Deets-Digital-Hackathon-/main/Hackthon%20Train%20Datac.csv")

#print(df.shape)
#print(df.info())
#print(round(df.describe().T,3))
#print(df.columns)

## Convert Bi-weekly to Bi_weekly
df['TERMFREQUENCY'].replace("Bi-weekly",'Bi_weekly',inplace = True)
#print(df.head())

# Segregate numerical and categorical features
num_features = [fea for fea in df.columns if df[fea].dtypes != "O"]
categorical_features = [fea for fea in df.columns if df[fea].dtypes == "O"]
# Check the null values
#print(df.isna().sum())
# check null value in BALANCE Feature
#print(df[df['BALANCE'].isna()][:5])

df.drop(df[df['BALANCE'].isna()].index, axis = 0, inplace = True)
df["PASTDUEAMOUNT"].fillna(df['PASTDUEAMOUNT'].median(),inplace = True)
df['PASTDUEAMOUNT'].isna().sum()
df['INSTALLMENTAMOUNT'].fillna(df['INSTALLMENTAMOUNT'].median(),inplace = True)
df['COLLATERALTYPE'].fillna(df['COLLATERALTYPE'].median(),inplace = True)
df['CREDITLIMIT'].fillna(df['CREDITLIMIT'].median(),inplace = True)
## for DATECLOSED we will replace null value with mode
df['DATECLOSED'].fillna(df['DATECLOSED'].mode()[0],inplace = True)
df['DATEOPENED'].fillna(df['DATEOPENED'].mode()[0],inplace = True)
df['REPAYMENTTENURE'].fillna(df['REPAYMENTTENURE'].mode()[0],inplace = True)
df['HIGHCREDIT'].fillna(df['HIGHCREDIT'].median(),inplace = True)
df['COLLATERALVALUE'].fillna(df['COLLATERALVALUE'].median(),inplace = True)
df['TERMFREQUENCY'].fillna(df['TERMFREQUENCY'].mode()[0],inplace = True)
df['SELF_TRADE'].fillna(df['SELF_TRADE'].mode()[0],inplace = True)
df['SECTOR'].fillna(df['SECTOR'].mode()[0],inplace = True)

# print(df.isna().sum())
# Statistical Analysis
#Check the correlation
# print(df.corr().T)

X = df.drop(columns = ['Defaulter','DATECLOSED','DATEOPENED','DOB','TERMFREQUENCY','SELF_TRADE','SECTOR'])
y = df['Defaulter']
df2 = df.copy()
X1 = df2.drop(columns = ['Defaulter','DATECLOSED','DATEOPENED','DOB','TERMFREQUENCY','SELF_TRADE','SECTOR','BALANCE'])

## List of Top 10 features with their importance value
ordered_rank_features=SelectKBest(score_func=chi2,k=9)
ordered_rank_features.fit(X1,y)
ordered_rank_features.fit(X1,y)
ordered_rank_features.scores_
df_scores=pd.DataFrame(ordered_rank_features.scores_,columns=["Scores"])
dfcolumns=pd.DataFrame(X1.columns)
features_rank=pd.concat([dfcolumns,df_scores],axis=1)
features_rank.columns=["features","score"]
round(features_rank.sort_values(by = ['score'],ascending=False)),3

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)
# print(X_train.shape,y_train.shape)
# print(X_test.shape,y_test.shape)

# Logistic Regression Model Training
classifier=LogisticRegression()
parameter = {'penalty':['l1','l2','elasticnet'],'C':[1,2,3,4,5,6,10,20,30,40,50],'max_iter':[100,200,300]}
classifier_regressor_bal=GridSearchCV(classifier,param_grid=parameter,scoring='accuracy',cv=5)

# Standarizing or Feature Scaling
classifier_regressor_bal.fit(X_train,y_train)

# Prediction
y_bal_pred = classifier_regressor_bal.predict(X_test)

# Accuracy
bal_score = accuracy_score(y_bal_pred,y_test)

# Confusion matric
conf_mat_bal=confusion_matrix(y_bal_pred,y_test)
true_positive = conf_mat_bal[0][0]
false_positive = conf_mat_bal[0][1]
false_negative = conf_mat_bal[1][0]
true_negative = conf_mat_bal[1][1]

bal_Precision = true_positive/(true_positive+false_positive)
bal_recall = true_positive/(true_positive+false_negative)
F1_Score_bal = 2*(bal_recall * bal_Precision) / (bal_recall + bal_Precision)
auc = roc_auc_score(y_bal_pred, y_test)


# Classification Report

print("ACCURACY:",bal_score)
print("Precision",bal_Precision)
print("Recall",bal_recall)
print("F1-Score:",F1_Score_bal)
print("Area under curve of original dataset",auc)
print("\nclassification_report ",classification_report(y_bal_pred,y_test))