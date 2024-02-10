#!/usr/bin/env python
# coding: utf-8

# In[83]:


# Basic Exploratory Data Analysis

# iMPORTING Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[84]:


df = pd.read_csv('Jamboree_Admission.csv')


# In[85]:


df.head()


# In[86]:


# cheking for null values 

df.isna().sum()


# In[87]:


# There are no null values in the dataset


# In[88]:


df.shape


# In[89]:


# The above dataset contains 500 rows and 9 columns


# In[90]:


# Cheking for unique values in each datset
df.nunique()


# In[91]:


df.info()


# In[92]:


# As, University Rating, SOP, LOR and Research values are very small, we can consider them as categorical variables.


# In[95]:


## datatypes of the dataset
df.dtypes


# In[96]:


#dropping unwanted column "Serial No" and deciding the target variable 


# In[97]:


df.drop(columns = 'Serial No.', inplace = True)


# In[98]:


df.head()


# In[99]:


# Weights/Independent Variable - GRE Score, TOEFL Score, University Raing, SOP, LOR, CGPA, Research
# Target Variable - Chance of Admit

df.columns = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'CGPA', 'Research', 'Chance_of_Admit' ]


# In[100]:


df.head()


# In[101]:


df.dtypes


# In[102]:


# Checking the correlation among all variables 


# In[103]:


sns.heatmap(df.corr(), annot = True, cmap='Blues')
plt.show()


# In[206]:


# Checking for outliers in the dataset

def detect_outliers(data):
    len_before = len(data)
    Q1 = np.percentile(data,25)
    Q3 = np.percentile(data,75)
    IQR = Q3 - Q1
    upperbound = Q3 + 1.5*IQR
    lowerbound = Q1 - 1.5*IQR
    if lowerbound < 0:
        lowerbound = 0
        
    len_after = len(data[data>lowerbound & data<upperbound])
    return np.round(((len_before - len_after)/len_before)*100,2)


# In[207]:


for i in df.columns:
    print(f"Outliers in i is : {detect_outliers(df[i])}")


# In[105]:


# Outliers in the categorical data 
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.boxplot(y = df["Chance_of_Admit"], x = df["SOP"])
plt.subplot(2,2,2)
sns.boxplot(y = df["Chance_of_Admit"], x = df["LOR"])
plt.subplot(2,2,3)
sns.boxplot(y = df["Chance_of_Admit"], x = df["University_Rating"])
plt.subplot(2,2,4)
sns.boxplot(y = df["Chance_of_Admit"], x = df["Research"])
plt.show()


# In[ ]:


# Outliers exist every categorical data
# The above box plots also shows that all features ar positively correlated with the target variable


# In[106]:


# Checking for correlation of features with the target variable

for col in df.columns:
    print(col)
    plt.figure(figsize=(3,3))
    sns.jointplot(data = df,x = df[col], y = df["Chance_of_Admit"],kind="reg")
    plt.show()


# In[107]:


# Above correlation plot shows a linear relation between every feature in the dataset and the target variable - Chance of Admit


# In[187]:


## Descriptive analysis of numerical variables

df.describe()


# In[ ]:


# chances of admit is within 0 to 1 (no outliers observed).
# Range of GRE score is between 290 to 340.
# Range of TOEFL score is between 92 to 120.
# University rating , SOP and LOR are distributed between range of 1 to 5.
# CGPA range is between 6.8 to 9.92


# In[202]:


## Graphical analysis

# Distribution of numerical variables

# sns.distplot(df["TOEFL_Score"])
# sm.qqplot(df["TOEFL_Score"],fit=True, line="45")
# plt.show()
# plt.figure(figsize=(14,5))
# sns.boxplot(y = df["Chance_of_Admit"], x = df["TOEFL_Score"])

plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.distplot(df["GRE_Score"])
plt.subplot(1,3,2)
sns.distplot(df["TOEFL_Score"])
plt.subplot(1,3,3)
sns.distplot(df["CGPA"])

plt.show()


# In[204]:


# Distribution of categorical variables

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.histplot(df["University_Rating"])
plt.subplot(2,2,2)
sns.histplot(df["LOR"])
plt.subplot(2,2,3)
sns.histplot(df["SOP"])
plt.subplot(2,2,4)
sns.histplot(df["Research"])


# In[205]:


sns.pairplot(df,y_vars = ["Chance_of_Admit"])
plt.title("Pair plot Chance of admit vs all the features")
plt.show()


# In[108]:


# Linear Regression


# In[109]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[110]:


# Independent variables - X
# Dependent/Target variable - y

X = df.drop(columns = 'Chance_of_Admit', axis = 1)
y = df['Chance_of_Admit'].values


# In[113]:


# Standardizing the dataset using StandardScaler 
scaler = StandardScaler()


# In[114]:


x = scaler.fit_transform(X)


# In[115]:


# Spliting the data into train and test data

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)


# In[182]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[117]:


# Fitting the model
linear = LinearRegression()


# In[118]:


linear.fit(X_train, y_train)


# In[119]:


# Checking the r2_score on train and test data
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# In[208]:


# r2_score on training data
y_pred = linear.predict(X_train)
r2score_train = r2_score(y_train,y_pred)
r2score_train


# In[210]:


# r2_score on test data
r2score_test = r2_score(y_test,linear.predict(X_test))
r2score_test


# In[213]:


# Calculating feature coefficients and intercepts
ws = pd.DataFrame(linear.coef_.reshape(1,-1),columns=df.columns[:-1])
ws["Intercept"] = linear.intercept_
ws


# In[214]:


def AdjustedR2score(R2,n,d):
    return 1-(((1-R2)*(n-1))/(n-d-1))


# In[221]:


# Checking the model metrics

# Training data performance checked

y_pred = linear.predict(X_train)
print("MSE:",mean_squared_error(y_train,y_pred)) # MSE
print("RMSE:",np.sqrt(mean_squared_error(y_train,y_pred))) #RMSE
print("MAE :",mean_absolute_error(y_train,y_pred) ) # MAE
print("r2_score:",r2_score(y_train,y_pred)) # r2score
print("Adjusted R2 score :", AdjustedR2score(r2_score(y_train,y_pred),len(X),X.shape[1]))


# In[222]:


# Test Performance checked

y_pred = linear.predict(X_test)
print("MSE:",mean_squared_error(y_test,y_pred)) # MSE
print("RMSE:",np.sqrt(mean_squared_error(y_test,y_pred))) #RMSE
print("MAE :",mean_absolute_error(y_test,y_pred) ) # MAE
print("r2_score:",r2_score(y_test,y_pred)) # r2score
print("Adjusted R2 score :", AdjustedR2score(r2_score(y_test,y_pred),len(X),X.shape[1]))


# In[137]:


# Checking the assumptions of linear regression

# 1. multicollinearity check

vif = []
for i in range(X_train.shape[1]):
    vif.append(variance_inflation_factor(exog = X_train, exog_idx = i))
vif


# In[138]:


pd.DataFrame({"coeff_name":X.columns, "vif":np.round(vif,2)})


# In[139]:


# From above analysis, we can say that since the vif_score of every feature is < 5, multicollinearity does not exist.


# In[223]:


# 2. Normality check of residuals

y_pred = linear.predict(X_train)
residual = y_train - y_pred
sm.qqplot(residual, line = '45')
plt.show()


# In[228]:


y_pred = linear.predict(X_test)
residual = y_test - y_pred
sm.qqplot(residual, line = '45')
plt.show()


# In[229]:


# the above plot shows that residuals are not normally distributed


# In[230]:


# 3. Test for Homoscedasticity

y_train_pred = linear.predict(X_train)
sns.scatterplot(x = y_pred,y = residual)
plt.xlabel('y_predicted')
plt.ylabel('residuals')
plt.axhline(y=0)
plt.title('y_predicted vs residuals check for Homoscedasticity')
plt.show()


# In[231]:


# the above analysis for homoscadesticity show that the spread for residuals is evenly distributed


# In[233]:


# 4. Linearity check betweeen dependent and independent variables
sns.pairplot(df,y_vars = ["Chance_of_Admit"])
plt.show()


# In[146]:


# The above plots show a linear relationship between the dependent and the target variable.


# In[147]:


# Model Regularization

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[148]:


# 1. Lasso Regression - L1 Regularization

lasso_regressor = Lasso()
parameters = {'alpha': [1,2,3,5,10,20,30,40,50,60,70,80,90,100]}


# In[149]:


lassocv =  GridSearchCV(lasso_regressor, param_grid = parameters, scoring = "neg_mean_squared_error", cv = 5)
lassocv.fit(X_train, y_train)


# In[150]:


print(lassocv.best_params_)


# In[151]:


## gives the best/optimal alpha value as 1


# In[152]:


lassocv.best_score_


# In[153]:


lasso_pred = lassocv.predict(X_test)


# In[154]:


sns.distplot((lasso_pred - y_test), kde = True)
plt.show()


# In[155]:


# The variance is too less within -0.4 to +0.4


# In[170]:


lasso_score = r2_score(lasso_pred,y_test)


# In[234]:


lasso_score


# In[158]:


# 2. Ridge regression - L1 regularization
from sklearn.linear_model import Ridge


# In[159]:


ridge_regressor = Ridge()


# In[160]:


parameters = {'alpha': [1,2,3,5,10,20,30,40,50,60,70,80,90,100]}


# In[162]:


ridgecv = GridSearchCV(ridge_regressor, param_grid = parameters, scoring = "neg_mean_squared_error", cv = 5)
ridgecv.fit(X_train, y_train)


# In[163]:


print(ridgecv.best_params_)


# In[164]:


ridgecv.best_score_


# In[166]:


ridge_pred = ridgecv.predict(X_test)
sns.distplot((ridge_pred - y_test), kde = True)
plt.show()


# In[167]:


# The variance is too less within -0.2 to +0.3


# In[169]:


ridge_score = r2_score(ridge_pred,y_test)
ridge_score


# In[239]:


# Checking for the errors after regularization

# Lasso Regression
LassoModel = Lasso(alpha=1)
LassoModel.fit(X_train , y_train)
trainR2 = LassoModel.score(X_train,y_train)
testR2 = LassoModel.score(X_test,y_test)


# In[240]:


(trainR2,testR2)


# In[248]:


Lasso_Model_coefs = pd.DataFrame(LassoModel.coef_.reshape(1,-1),columns=df.columns[:-1])
Lasso_Model_coefs["Intercept"] = LassoModel.intercept_
Lasso_Model_coefs


# In[252]:


y_pred = LassoModel.predict(X_test)

print("MSE:",mean_squared_error(y_test,y_pred)) # MSE
print("RMSE:",np.sqrt(mean_squared_error(y_test,y_pred))) #RMSE
print("MAE :",mean_absolute_error(y_test,y_pred) ) # MAE
print("r2_score:",r2_score(y_test,y_pred)) # r2score
print("Adjusted R2 score :", AdjustedR2score(r2_score(y_test,y_pred),len(X),X.shape[1])) 


# In[249]:


RidgeModel = Ridge(alpha = 10)
RidgeModel.fit(X_train,y_train)
trainR2 = RidgeModel.score(X_train,y_train)
testR2 = RidgeModel.score(X_test,y_test)
(trainR2,testR2)


# In[250]:


RidgeModel_coefs = pd.DataFrame(RidgeModel.coef_.reshape(1,-1),columns=df.columns[:-1])
RidgeModel_coefs["Intercept"] = RidgeModel.intercept_
RidgeModel_coefs


# In[251]:


y_pred = RidgeModel.predict(X_test)

print("MSE:",mean_squared_error(y_test,y_pred)) # MSE
print("RMSE:",np.sqrt(mean_squared_error(y_test,y_pred))) #RMSE
print("MAE :",mean_absolute_error(y_test,y_pred) ) # MAE
print("r2_score:",r2_score(y_test,y_pred)) # r2score
print("Adjusted R2 score :", AdjustedR2score(r2_score(y_test,y_pred),len(X),X.shape[1]))


# In[253]:


# SUMMARY
#Linear Regression
y_pred = linear.predict(X_test)
LinearRegression_model_metrics = []
LinearRegression_model_metrics.append(mean_squared_error(y_test,y_pred)) # MSE
LinearRegression_model_metrics.append(np.sqrt(mean_squared_error(y_test,y_pred))) #RMSE
LinearRegression_model_metrics.append(mean_absolute_error(y_test,y_pred) ) # MAE
LinearRegression_model_metrics.append(r2_score(y_test,y_pred)) # r2score
LinearRegression_model_metrics.append(AdjustedR2score(r2_score(y_test,y_pred),len(X),X.shape[1]))  # adjusted R2 score

#Ridge Regression
y_pred = RidgeModel.predict(X_test)
RidgeModel_model_metrics = []
RidgeModel_model_metrics.append(mean_squared_error(y_test,y_pred)) # MSE
RidgeModel_model_metrics.append(np.sqrt(mean_squared_error(y_test,y_pred))) #RMSE
RidgeModel_model_metrics.append(mean_absolute_error(y_test,y_pred) ) # MAE
RidgeModel_model_metrics.append(r2_score(y_test,y_pred)) # r2score
RidgeModel_model_metrics.append(AdjustedR2score(r2_score(y_test,y_pred),len(X),X.shape[1]))  # adjusted R2 score

#Lasso Regression
y_pred = LassoModel.predict(X_test)
LassoModel_model_metrics = []
LassoModel_model_metrics.append(mean_squared_error(y_test,y_pred)) # MSE
LassoModel_model_metrics.append(np.sqrt(mean_squared_error(y_test,y_pred))) #RMSE
LassoModel_model_metrics.append(mean_absolute_error(y_test,y_pred) ) # MAE
LassoModel_model_metrics.append(r2_score(y_test,y_pred)) # r2score
LassoModel_model_metrics.append(AdjustedR2score(r2_score(y_test,y_pred),len(X),X.shape[1]))  # adjusted R2 score


# In[254]:


A = pd.DataFrame([LinearRegression_model_metrics,LassoModel_model_metrics,RidgeModel_model_metrics],columns=["MSE","RMSE","MAE","R2_SCORE","ADJUSTED_R2"],index = ["Linear Regression Model","Lasso Regression Model","Ridge Regression Model"])
A


# In[ ]:


# Insights , Feature Importance and Interpretations and Recommendations :

# - University Rating , SOP and LOR strength and research seem to be discrete random Variables , but also ordinal numeric data.

# - all the other features are numeric, ordinal and continuous.

# - No null values were present in data.

# - No Significant amount of outliers were found in data.

# - correlation heatmap shows a strong correlation between the GRE score, TOEFL score and CGPA with Change of admission.

# - University rating, SOP ,LOR and Research have comparatively slightly less correlated than other features.

# - Students having high GRE score , has higher probability of getting admission .

# - Students having high TOEFL score , has higher probability of getting admission .

# - the performance metrics are very similar for dataset fitted using linear model and after ridge regression

# Actionable Insights and Recommendations :

# - Awareness of CGPA and Research Capabilities : Seminars can be organised to increase the awareness regarding CGPA and Research Capablities to enhance the chance of admit.
# - GRE and TOEFL scores also play an important role as they are linearly distributed with the probability of getting admission.
# - Proper awareness and coaching for the above mentioned exams as well as maintaing good SOP and LOR can help the student get admission in a high ranked university.

