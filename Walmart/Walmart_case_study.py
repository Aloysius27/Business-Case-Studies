#!/usr/bin/env python
# coding: utf-8

# In[16]:


# importing necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, binom


# In[17]:


df = pd.read_csv('Walmart.csv')


# In[18]:


df.head()


# In[19]:


df.info()


# In[20]:


# Checking for total rows and columns in the dataset

df.shape


# In[21]:


# There are total 550068 rows and 10 columns in the entire dataset. 


# In[22]:


# checking for null/missing values
df.isna().sum()


# In[23]:


df.describe()


# In[24]:


# Average purchase done by a customer is 9263.968713 where the min purchase is 12.00 and max purchase is 23961.00


# In[25]:


sns.histplot(x = 'Age', data = df)
plt.show()


# In[26]:


# total percentage of male and female customers
round(df.Gender.value_counts(normalize = True)*100,2)


# In[27]:


# Observation –
# Above analysis show that Male customers purchase more than female customers.
# Around 75.31 % Males customers do purchasing in Walmart compared to Female which 
# is 24.69% only.


# In[28]:


# Business Analysis
# Average spending analysis of male and female customers

from scipy.stats import ttest_ind


# In[29]:


male_mean = df[df['Gender'] == 'M']['Purchase'].mean()
female_mean = df[df['Gender'] == 'F']['Purchase'].mean()


# In[30]:


male_mean, female_mean


# In[31]:


# Hypothesis testing to check if the spending is gender biased?
                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
male = df[df['Gender'] == 'M']['Purchase']
female = df[df['Gender'] == 'F']['Purchase']

t_stat, p_value = ttest_ind(male, female, alternative = 'less')


# In[32]:


p_value


# In[33]:


male


# In[34]:


df.Product_Category.nunique()


# In[35]:


# There are 20 unique product categories in this dataset.


# In[36]:


# Gender purchases based on age

sns.countplot(data = df, x = 'Age', hue = 'Gender', palette = 'coolwarm')
plt.xlabel('Age', fontweight = 'bold')
plt.ylabel('Total count', fontweight = 'bold')
plt.show()


# In[37]:


# Above analysis show that male customers in the age group of 26-35 do maximum spending compared to other age-groups.  
# Also, for both the genders, customers in the age-group of 26-35 spend the most.


# In[38]:


# No of customer purchases based on product type
round(df.Product_Category.value_counts(normalize = True)*100,2)


# In[39]:


# Product Categories 1,5 and 8 are the maximum purchases. 
# Thus, least popular product among the customers is Product_Category 9. 


# In[40]:


# Outlier detection using boxplot
sns.boxplot(data = df, x = 'Age', y = 'Purchase', hue = 'Gender', palette = 'coolwarm')
plt.xlabel('Age', fontweight = 'bold')
plt.ylabel('Purcahse amount', fontweight = 'bold')
plt.show()


# In[41]:


# Observations –
# Outliers are seen in both the plots, where maximum outliers are observed for purchases 
# of female customers of the age-group 55+ and the age-group 18-45. 


# In[81]:


sns.displot(data = female,  kind = 'kde')


# In[82]:


male.mean()


# In[83]:


np.random.choice(male, size = 10)


# In[84]:


# Distribution of the mean spending by MALE customers

# 1. Taking sample size = 100 and checking for 50 datapoints. 

bootstrap = []

for i in range(50):
    bootstrap_samples = np.random.choice(male, size = 100)
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap.append(bootstrap_mean)

sns.histplot(bootstrap, bins = 10, kde = True)
plt.title('Sample size = 100 and 50 datapoints', fontweight = 'bold')
plt.xlabel('Purchase Amount', fontweight = 'bold')
plt.show()


# In[85]:


# 2. Taking sample size = 100 and 500 datapoints

bootstrap_1 = []

for i in range(500):
    bootstrap_samples = np.random.choice(male, size = 100)
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_1.append(bootstrap_mean)
    
sns.histplot(bootstrap_1, bins = 10, kde = True)
plt.title('Sample size = 100 and 500 datapoints', fontweight = 'bold')
plt.xlabel('Purchase Amount', fontweight = 'bold')
plt.show()


# In[86]:


# 3. Taking sample size = 100 and 10,000 datapoints

bootstrap_2 = []

for i in range(10000):
    bootstrap_samples = np.random.choice(male, size = 100)
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_2.append(bootstrap_mean)
    
sns.histplot(bootstrap_2, bins = 10, kde = True)
plt.title('Sample size = 100 and 10000 datapoints', fontweight = 'bold')
plt.xlabel('Purchase Amount', fontweight = 'bold')
plt.show()


# In[52]:


# 4. Taking sample size = 200 and 20,000 datapoints

bootstrap_3 = []

for i in range(20000):
    bootstrap_samples = np.random.choice(male, size = 200)
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_3.append(bootstrap_mean)
    
sns.histplot(bootstrap_3, bins = 10, kde = True)
plt.title('Sample size = 200 and 20000 datapoints', fontweight = 'bold')
plt.xlabel('Purchase Amount', fontweight = 'bold')
plt.show()


# In[88]:


# Observations –
#Variance (spread of data) decreases as sample size and datapoints increase. 


# In[89]:


male_std = round(male.std(),2)


# In[90]:


female_std = round(female.std(),2)


# In[92]:


# Distribution of the mean spending of FEMALE customers

bootstrap_female = []

for i in range(50):
    bootstrap_samples = np.random.choice(female, size = 100)
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_female.append(bootstrap_mean)
    
sns.histplot(bootstrap_female, bins = 10, kde = True)
plt.title('Sample size = 100 and 50 datapoints', fontweight = 'bold')
plt.xlabel('Purchase Amount', fontweight = 'bold')
plt.show()


# In[93]:


bootstrap_1_female = []

for i in range(500):
    bootstrap_samples = np.random.choice(female, size = 100)
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_1_female.append(bootstrap_mean)
    
sns.histplot(bootstrap_1_female, bins = 10, kde = True)
plt.title('Sample size = 100 and 500 datapoints', fontweight = 'bold')
plt.xlabel('Purchase Amount', fontweight = 'bold')
plt.show()


# In[94]:


bootstrap_2_female = []

for i in range(10000):
    bootstrap_samples = np.random.choice(female, size = 100)
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_2_female.append(bootstrap_mean)
    
sns.histplot(bootstrap_2_female, bins = 10, kde = True)
plt.title('Sample size = 100 and 10000 datapoints', fontweight = 'bold')
plt.xlabel('Purchase Amount', fontweight = 'bold')
plt.show()


# In[95]:


bootstrap_3_female = []

for i in range(20000):
    bootstrap_samples = np.random.choice(female, size = 200)
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_3_female.append(bootstrap_mean)
    
sns.histplot(bootstrap_3_female, bins = 10, kde = True)
plt.title('Sample size = 200 and 20000 datapoints', fontweight = 'bold')
plt.xlabel('Purchase Amount', fontweight = 'bold')
plt.show()


# In[96]:


# Observations –
# Comparison between both the distributions infer that the mean spending of Male 
# customers is more compared to the female customers.


# In[97]:


# As the variance is less for sample size 200 and 20,000 datapoints, we calculate the 
# confidence intervals for 90%, 95% and 99% confidence levels for the specified 
# sample size and sample mean.


# In[98]:


# Sample_mean for sample size = 200
sample_mean_male = sum(bootstrap_3)/len(bootstrap_3)
round(sample_mean_male,2)


# In[99]:


sample_mean_female = sum(bootstrap_3_female)/len(bootstrap_3_female)
round(sample_mean_female,2)


# In[100]:


# From the above sample means, we can say that the average amount spent by male 
# customers is 9434.45 and the average amount spent by female customers is 8730.19. 


# In[101]:


# Validating the difference in the mean spending with different confidence interval–

# For Male customers

# 90% Confidence Level
x1 = np.percentile(bootstrap_3, 5)
x2 = np.percentile(bootstrap_3, 95)


# In[102]:


x1, x2


# In[103]:


# With 90% Confidence Interval, the mean spending of male customers lie in the range (8848.144, 10024.076).


# In[104]:


# 95% Confidence Level
x1 = np.percentile(bootstrap_3, 2.5)
x2 = np.percentile(bootstrap_3, 97.5)


# In[105]:


x1, x2


# In[106]:


# With 95% Confidence Interval, the mean spending of male customers lie in the range (8741.456, 10133.311). 


# In[107]:


# 99% Confidence Level
x1 = np.percentile(bootstrap_3, 0.5)
x2 = np.percentile(bootstrap_3, 99.5)


# In[108]:


x1, x2


# In[109]:


# With 99% Confidence Interval, the mean spending of male customers lie in the range  (8513.545, 10380.782). 


# In[110]:


# For Female customers

# 90% Confidence Level
y1 = np.percentile(bootstrap_3_female, 5)
y2 = np.percentile(bootstrap_3_female, 95)


# In[111]:


y1,y2


# In[112]:


# With 90% Confidence Interval, the mean spending of female customers lie in the range (8174.314, 9292.295). 


# In[113]:


# 95% Confidence Level
y1 = np.percentile(bootstrap_3_female, 2.5)
y2 = np.percentile(bootstrap_3_female, 97.5)


# In[114]:


y1,y2


# In[115]:


# With 95% Confidence Interval, the mean spending of female customers lie in the range  (7867.459, 9603.082). 


# In[116]:


# 99% Confidence Level
y1 = np.percentile(bootstrap_3_female, 0.5)
y2 = np.percentile(bootstrap_3_female, 99.5)


# In[117]:


y1,y2


# In[118]:


# With 99% Confidence Interval, the mean spending of female customers lie in the range (7867.4595, 9603.917). 


# In[119]:


# Observations –
# Confidence intervals are found to be overlapping. This concludes that there is no 
# significant difference in the mean spending by male and female customers. 


# In[121]:


# Distribution of mean spending based on Marital Status

# 1 - Married
# 0 - Unmarried


um = df[df["Marital_Status"] == 0]['Purchase']
m = df[df["Marital_Status"] ==1]['Purchase']

# MARRIED CUSTOMERS

bootstrap_1_m = []
for i in range(50):
    bootstrap_samples = np.random.choice(m, size = 100) 
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_1_m.append(bootstrap_mean)
    
sns.histplot(bootstrap_1_m, bins = 10, kde = True)
plt.title('Sample size = 100 and 50 datapoints', fontweight = 'bold')
plt.xlabel('Purchase Amount', fontweight = 'bold')
plt.show()


# In[122]:


bootstrap_2_m = []
for i in range(500):
    bootstrap_samples = np.random.choice(m, size = 100) 
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_2_m.append(bootstrap_mean)
    
sns.histplot(bootstrap_2_m, bins = 10, kde = True)
plt.title('Sample size = 100 and 500 datapoints', fontweight = 'bold')
plt.xlabel('Purchase Amount', fontweight = 'bold')
plt.show()


# In[123]:


bootstrap_3_m = []

for i in range(10000):
    bootstrap_samples = np.random.choice(m, size = 200) 
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_3_m.append(bootstrap_mean)
    
sns.histplot(bootstrap_3_m, bins = 10, kde = True)
plt.title('Sample size = 10000 and 200 datapoints', fontweight = 'bold')
plt.xlabel('Purchase Amount', fontweight = 'bold')
plt.show()


# In[124]:


bootstrap_4_m = []

for i in range(20000):
    bootstrap_samples = np.random.choice(m, size = 200) 
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_4_m.append(bootstrap_mean)
    
sns.histplot(bootstrap_4_m, bins = 10, kde = True)
plt.title('Sample size = 20000 and 200 datapoints', fontweight = 'bold')
plt.xlabel('Purchase Amount', fontweight = 'bold')
plt.show()


# In[125]:


# UNMARRIED CUSTOMERS

bootstrap_1_um = []
um = df[df["Marital_Status"] == 0]['Purchase']
m = df[df["Marital_Status"] ==1]['Purchase']

for i in range(10000):
    bootstrap_samples = np.random.choice(um, size = 200) 
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_1_um.append(bootstrap_mean)
    
sns.histplot(bootstrap_1_um, bins = 10, kde = True)
plt.title('Sample size = 100 and 50 datapoints', fontweight = 'bold')
plt.xlabel('Purchase Amount', fontweight = 'bold')
plt.show()


# In[126]:


bootstrap_2_um = []

for i in range(500):
    bootstrap_samples = np.random.choice(um, size = 100) 
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_2_um.append(bootstrap_mean)
    
sns.histplot(bootstrap_2_um, bins = 10, kde = True)
plt.title('Sample size = 100 and 500 datapoints', fontweight = 'bold')
plt.xlabel('Purchase Amount', fontweight = 'bold')
plt.show()


# In[127]:


bootstrap_3_um = []

for i in range(10000):
    bootstrap_samples = np.random.choice(um, size = 200) 
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_3_um.append(bootstrap_mean)
    
sns.histplot(bootstrap_3_um, bins = 10, kde = True)
plt.title('Sample size = 200 and 10000 datapoints', fontweight = 'bold')
plt.xlabel('Purchase Amount', fontweight = 'bold')
plt.show()


# In[128]:


bootstrap_4_um = []

for i in range(20000):
    bootstrap_samples = np.random.choice(um, size = 200) 
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_4_um.append(bootstrap_mean)
    
sns.histplot(bootstrap_4_um, bins = 10, kde = True)
plt.title('Sample size = 200 and 20000 datapoints', fontweight = 'bold')
plt.xlabel('Purchase Amount', fontweight = 'bold')
plt.show()


# In[129]:


def confidence(ms = marital_status(), r, n):
    
    if ms == m:
        bootstrap_m = []
        for i in range(r):
            bootstrap_samples = np.random.choice(ms, size = n) 
            bootstrap_mean = np.mean(bootstrap_samples)
            bootstrap_um.append(bootstrap_mean)
            
        sns.histplot(bootstrap_m, bins = 10, kde = True)
        plt.title('Sample size = 100 and 50 datapoints', fontweight = 'bold')
        plt.xlabel('Purchase Amount', fontweight = 'bold')
        plt.show()
    else:
        bootstrap_um = []
        for i in range(r):
            bootstrap_samples = np.random.choice(ms, size = n) 
            bootstrap_mean = np.mean(bootstrap_samples)
            bootstrap_um.append(bootstrap_mean)
        
        sns.histplot(bootstrap_um, bins = 10, kde = True)
        plt.title('Sample size = 100 and 50 datapoints', fontweight = 'bold')
        plt.xlabel('Purchase Amount', fontweight = 'bold')
        plt.show()
            
    


# In[ ]:


def marital_status(input()):
    if m == m:
        m = df[df["Marital_Status"] ==1]['Purchase']
        return m
    else:
        um = df[df["Marital_Status"] == 0]['Purchase']
        return um


# In[ ]:


confidence(m,500,100)


# In[ ]:


# Validating the difference in the mean spending based on Marital Status

sample_mean_married = sum(bootstrap_4_m)/len(bootstrap_4_m)
round(sample_mean_married,2)


# In[ ]:


sample_mean_unmarried = sum(bootstrap_4_um)/len(bootstrap_4_um)
round(sample_mean_unmarried,2)


# In[206]:


# 90% Confidence Level
m1 = np.percentile(bootstrap_4_m, 5)
m2 = np.percentile(bootstrap_4_m, 95)


# In[207]:


m1,m2


# In[208]:


# 95% Confidence Level
m1 = np.percentile(bootstrap_4_m, 2.5)
m2 = np.percentile(bootstrap_4_m, 97.5)


# In[209]:


m1,m2


# In[211]:


# 99% Confidence Level
m1 = np.percentile(bootstrap_4_m, 0.5)
m2 = np.percentile(bootstrap_4_m, 99.5)


# In[212]:


m1,m2


# In[213]:


# 90% Confidence Level
um1 = np.percentile(bootstrap_4_um, 5)
um2 = np.percentile(bootstrap_4_um, 95)


# In[214]:


um1,um2


# In[215]:


# 95% Confidence Level
um1 = np.percentile(bootstrap_4_um, 2.5)
um2 = np.percentile(bootstrap_4_um, 97.5)


# In[216]:


um1,um2


# In[217]:


# 99% Confidence Level
um1 = np.percentile(bootstrap_4_um, 0.5)
um2 = np.percentile(bootstrap_4_um, 99.5)


# In[218]:


um1,um2


# In[ ]:


# Observations –
# Confidence intervals are found to be overlapping. This concludes that there is no 
# significant difference in the mean spending by married and unmarried customers.


# In[ ]:


# Recommendations
# 1. Men spent more money than women, so company should focus on retaining the male customers and getting more male customers.
# 2. Product_Category - 1, 5 & 8 have highest purchasing frequency. It means these are the products in these categories are liked more by customers. Company can 
# focus on selling more of these products or selling more of the products which 
# are purchased less.
# 3. Customers in the age 18-45 spend more money than the others, so company should focus on acquisition of customers who are in the age 18-45
# 4. Male customers living in City_Category C spend more money than other male customers living in B or C, Selling more products in the City_Category C will 
# help the company increase the revenue


# In[221]:


categorical_cols = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category']
df[categorical_cols].melt().groupby(['variable', 'value'])[['value']].count()/len(df)


# In[222]:


amt_df = df.groupby(['User_ID', 'Gender'])[['Purchase']].sum()
amt_df = amt_df.reset_index()
amt_df


# In[223]:


df.head()


# In[224]:


df.User_ID.nunique()


# In[225]:


categorical_cols = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years',
                    'Marital_Status', 'Product_Category']
df[categorical_cols].melt().groupby(['variable', 'value'])[['value']].count()/len(df)


# In[230]:


#outlier detection
sns.boxplot(data = df, y = 'Purchase', x = 'Age', hue = 'Gender' )
plt.show()


# In[232]:


sns.boxplot(data = df, y = 'Purchase', x = 'Age', hue = 'Marital_Status' )
plt.show()


# In[236]:


sns.countplot(data = df, x = 'Product_Category')
plt.show()


# In[246]:


round(df.Product_Category.value_counts(normalize = True)*100,2)


# In[247]:


attrs = ['Gen
         der', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category']
sns.set_style("white")

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20, 16))
fig.subplots_adjust(top=1.3)
count = 0
for row in range(3):
    for col in range(2):
        sns.boxplot(data=df, y='Purchase', x=attrs[count], ax=axs[row, col], palette='Set3')
        axs[row,col].set_title(f"Purchase vs {attrs[count]}", pad=12, fontsize=13)
        count += 1
plt.show()

plt.figure(figsize=(10, 8))
sns.boxplot(data=df, y='Purchase', x=attrs[-1], palette='Set3')
plt.show()


# In[256]:


fig = plt.figure(figsize = (12,8))

plt.subplot(2,3,1)
sns.boxplot(data = df, x = 'Gender', y = 'Purchase')
plt.title('Purchase vs Gender', fontweight = 'bold')

plt.subplot(2,3,3)
sns.boxplot(data = df, x = 'City_Category', y = 'Purchase')
plt.title('Purchase vs City_Category', fontweight = 'bold')

plt.subplot(2,3,5)
sns.boxplot(data = df, x = 'Marital_Status', y = 'Purchase')
plt.title('Purchase vs Marital_Status', fontweight = 'bold')

plt.show()


# In[287]:


# Checking the variation in the mean/ average of customers using Hypothesis Testing.
# Using ANOVA, we can find the variation in the mean spending for 5% significance.
# Spending based on Marital_Status

# H0: All means are similar
# Ha: Means are different.
# Alpha = 0.05 (95% Confidence Level)

married = df[df['Marital_Status'] == 0]['Purchase']
unmarried = df[df['Marital_Status'] == 1]['Purchase']


# In[288]:


from scipy.stats import f_oneway


# In[289]:


f_stats, p_value = f_oneway(married, unmarried)


# In[290]:


p_value


# In[297]:


# Spending based on Age

age18_25 = df[df['Age'] == '18-25']['Purchase']
age26_35 = df[df['Age'] == '26-35']['Purchase']
age36_45 = df[df['Age'] == '36-45']['Purchase']


# In[ ]:


# Since, p_value > alpha, We fail to reject the Null Hypothesis (H0).
# Hence, we can conclude that with a 95% Confidence Level, there is no difference in the 
# average spending of customers based on their Marital Status.


# In[ ]:


# 2. Spending based on Age
# H0: All means are similar
# Ha: Means are different.
# Alpha = 0.05 (95% Confidence Level)


# In[298]:


f_stats, p_value = f_oneway(age18_25, age26_35, age36_45)


# In[299]:


p_value


# In[ ]:


# Since, p_value < alpha, We reject the Null Hypothesis (H0).
# Hence, we can conclude that with a 95% Confidence Level, there is a significant
# difference in the average spending of customers based in the age-groups 18-25, 26-35, 36-45.


# In[ ]:


# Recommendations –
# Walmart must focus to retain the customers in these age-groups by providing discount
# offers and special contests to attract more customers belonging to these age-groups to 
# increase their revenue

