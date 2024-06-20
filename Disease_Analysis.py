#!/usr/bin/env python
# coding: utf-8

# ###### In this Notebook we are going to understand the data of patients, discuss information we gain from it and build Machine Learning model

# In[1]:


get_ipython().system('pip install xgboost')


# In[2]:


#Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy.stats import stats
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report


# In[3]:


#import data
df = pd.read_csv("disease_dataset.csv")
df.head()


# In[4]:


#to understand the columns present in the data
print(df.columns)


# # 1. Data Exploration

# ##### a. Perform initial data exploration on the given dataset and present your findings.

# In[6]:


#exploring the data further
df.info()


# In[7]:


print('Number of Columns: {0}'.format(len(df.columns)))

df.columns


# In[8]:


df.shape


# ##### Checking for dupliacted rows

# In[9]:


df.duplicated().sum()


# There are no duplicate records present in the given data

# ##### Checking for unique values

# In[10]:


df.nunique()


# In[11]:


for column in df.columns:
    print('----------------------------------------------------------------------')
    
    print('Unique Values of {0}'.format(column))
    
    print('----------------------------------------------------------------------')
    
    print(df[column].unique())
    
    print('-----------------------------------------------------------------------')
    
    print(df[column].value_counts())


# ##### To find the Distribution of the features

# In[12]:


#to check only categorical columns
df.select_dtypes(include="object")


# In[13]:


#to check numeric columns
df.select_dtypes(exclude="object")


# ##### To understand proportion of death vs Censored vs censored due to transplant Gender wise
# 

# In[14]:


####Countplot
plt.figure(figsize=(8, 5))

sns.set_theme(context='paper', font_scale=1.2)

countplot = sns.countplot(x='FinalStatus', hue='Gender',  palette="Set1", data=df)

countplot.set_ylabel("Count")

countplot.set_title("Value Count of Final Status in terms of Gender")

plt.tight_layout()


# From the graph we can see that overall count of death, censored and censored due to transplant is high if "Female" population as compared to "Male" population

# ##### Histogram of raw data before imputing missing values

# In[15]:


df.hist(bins=25, figsize=(20,15))

plt.show()


# In[16]:


df.describe()


# Average mean for each feature fluctuates a lot fromeach other

# ##### Missing Values check

# In[17]:


missing_value_count = df.isnull().sum()
missing_value_percentage = df.isnull().mean()*100 
missing_value_count_percentage_df = pd.concat([missing_value_count, missing_value_percentage], axis = 1, keys = ['Count', 'Percentage'])
missing_value_count_percentage_df = missing_value_count_percentage_df.sort_values(by = ['Count'])
missing_value_count_percentage_df


# Around 7 features have 25% missing values present and 2 features have more than 30% missing values present in it

# In[18]:


#To dealwith this we use mean/median imputation for continuous features
#As LipidProfile1 & 2 has more than 30% missing values we willimpute it with median and all other with mean
#Use mode imputer for categorcal features


# In[19]:


df.columns


# In[20]:


df.head(2)


# In[21]:


mean_impute = SimpleImputer(strategy="mean")
med_impute = SimpleImputer(strategy="median")
mode_impute = SimpleImputer(strategy="most_frequent")
feature_list1 = ['ProthrombinLevel', 'PlateletsCount', 'AlkalinePhosphateLevel', 'SGOTLevel', 'CuLevel']
feature_list2 = ['LipidProfile1', 'LipidProfile2']

df[feature_list1] =mean_impute.fit_transform(df[feature_list1])
df[feature_list2] =med_impute.fit_transform(df[feature_list2])

df.head()


# In[22]:


df['StageofDisease'].value_counts()


# In[23]:


df['StageofDisease'].isna().sum()


# In[25]:


categorical_features = ['StageofDisease','Medication', 'AscitesStatus', 'LiverSize', 'SpiderAngiomas']

for feature in categorical_features:
    df[feature] = mode_impute.fit_transform(df[[feature]])
    
    
#     mode_value = df[feature].mode()[0]  # Calculate mode
#     df[feature] = df[feature].fillna(mode_value)
# print(df.isna().sum())


# In[26]:


df.head()


# all the missing values have been removed, now to check the distrubution of each feature again

# In[27]:


df.hist(bins=25, figsize=(20,15))

plt.show()


# From the above graph we can see that Age and proteinlevel are approximately normally distributed, and all other are slight skewed

# In[28]:


#To understand it in more proper way
for col in df[feature_list1]:
    sns.displot(data=df, x=df[col], kde=True)
    


# Almost all these can be considered as rightly skewed

# In[29]:


#To understand Target variable 
df['FinalStatus'].value_counts()


# In[30]:


plt.pie(df['FinalStatus'].value_counts(),labels=df['FinalStatus'].value_counts().index,autopct='%1.2f%%', colors=sns.color_palette('Set2'))


# More than 50% data is sensored

# In[31]:


df['PlateletsCount'].min(),df['PlateletsCount'].max()


# In[32]:


# def probability_plot(feature):
#     print(f'{feature}')
#     plt.figure(figsize=(12,4))
#     plt.subplot(1,2,1)  #1st plot
#     plt.title(feature)
#     df[feature].hist()
#     plt.subplot(1,2,2)  #2nd plot
#     stats.probplot(df[feature], dist='norm', plot=pylab)
#     plt.tight_layout
#     plt.show()#to check again


# In[33]:


# probability_plot('Age')


# ##### To check the correlation between features

# In[34]:


def corr_heatmap(dataset, titles=''): 
    plt.figure(figsize = (10,7))
    sns.set_theme(context='paper', style='dark')
    plt.title(titles)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(dataset.corr(),annot=True,cmap=cmap)


# In[35]:


corr_heatmap(df.iloc[:,1:].corr(),'Correlation of Features')


# In[36]:


#skewness and kurtosis
df.skew()


# In[37]:


df.kurtosis()


# #### b. Outlier detection using Graphs

# In[38]:


plt.figure(figsize=(15,6),dpi=80)  
plt.title("Boxplot to check extreme observations", fontsize=12)
sns.boxplot(data = df, palette="Set3")
plt.ylim(-5, 500)
plt.xticks(rotation=45)


# From the above graph we can see that CuLevel, SGOTlevel,LipidProfile1,LipidProfile2 have extreme observations present hence there are chnaces of outeliers present in these features

# #### To deal with these outliers we use IQR method

# In[39]:


numeric_columns= df.select_dtypes(include="number")
numeric_columns
    


# In[40]:


def outlier_handle(df,columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
#             df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            return df


# In[41]:


feature_list2


# In[42]:


feature_list1


# In[43]:


columns_to_check = ['LipidProfile1', 'LipidProfile2','ProthrombinLevel',
 'PlateletsCount',
 'AlkalinePhosphateLevel',
 'SGOTLevel',
 'CuLevel']


# In[44]:


df_new = outlier_handle(df,columns_to_check)
df_new


# In[45]:


df_new.reset_index(drop=True,inplace=True)
df_new.head()


# In[46]:


df_new.shape


# tried removing the outliers but data size reduced from 418 to 369 to avoid this data loss, cap the values.

# In[ ]:





# In[47]:


# Check the resulting DataFrame
df_new.describe()


# ## 2. Data Pre-processing

# ###### a. Remove duplicates if any

# In[48]:


df_new.duplicated().sum()


# In[49]:


#there are no duplicate values present


# ###### b. fix null values problem

# This has already been done above where we imputed values with mean,mode,and median

# In[50]:


sns.countplot(data=df_new, x="FinalStatus", order=df['FinalStatus'].value_counts().index)


# In[51]:


df_new['FinalStatus'].value_counts()


# There is class imbalance present in the data

# we can use SMOTE, randomOversampler etc.

# In[52]:


df_new.head(2)


# In[53]:


#To remove class imbalance define X,y 
X = df_new.drop('FinalStatus', axis=1)
y = df_new['FinalStatus']

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['FinalStatus'])], axis=1)

print(df_resampled['FinalStatus'].value_counts())
df_resampled


# In[54]:


df_resampled.shape


# Here we used Randomoversampler, to avoid noise and incorrect real time patient data we did not use SMOTE.

# ## 3. Statistical Analysis 

# ##### a.Conduct a t-test to compare the mean ‘Bilirubin’ levels between patients with and without ‘Ascites’. What do the results indicate?

# In[55]:


df_resampled['AscitesStatus'].value_counts()


# In[56]:


df1 = df_resampled.copy() #resampled dataframe df1


# In[57]:


df['AscitesStatus'].value_counts()


# In[58]:


bilirubin_with_ascites = df1[df1['AscitesStatus'] == 'accumulation of fluid' ]['BilirubinLevel']
bilirubin_without_ascites = df1[df1['AscitesStatus'] == 'no fluid buildup' ]['BilirubinLevel']


# In[59]:


bilirubin_with_ascites


# In[60]:


bilirubin_without_ascites


# In[61]:


tstat, p_value = stats.ttest_ind(bilirubin_with_ascites, bilirubin_without_ascites)


# In[62]:


print(f"T-statistic: {tstat}")
print(f"P-value: {p_value}")

alpha = 0.05  # significance level
if p_value < alpha:
    print("The difference in mean Bilirubin levels between patients with and without Ascites is statistically significant.")
else:
    print("There is no significant difference in mean Bilirubin levels between patients with and without Ascites.")


# b.Conduct an ANOVA test to determine if there are significant differences in ‘Prothrombin’ levels among different ‘Stages’ of the disease, explain your analysis.

# In[63]:


df1['StageofDisease'].unique()


# In[64]:


df1


# In[65]:


#to encode the data using one hot encoding
df1.select_dtypes(include="object")


# In[66]:


encoded_df = pd.get_dummies(df1, columns=['Medication', 'Gender', 'AscitesStatus', 'LiverSize', 'SpiderAngiomas', 'FluidAccumulationSwelling', 'DiureticTherapy'])


# In[67]:


encoded_df


# In[68]:


df1 = df1.drop(columns=['Medication', 'Gender', 'AscitesStatus', 'LiverSize', 'SpiderAngiomas', 'FluidAccumulationSwelling', 'DiureticTherapy']) #to remove originalcolumns and add encoded columns


# In[69]:


df1.head()#14 columns


# In[70]:


encoded_df.isna().sum()


# In[71]:


df1.shape


# In[72]:


df1.columns


# In[73]:


encoded_df.columns


# In[74]:


df_combine = pd.concat([df1,encoded_df.iloc[:,14:]],axis=1)
df_combine


# In[75]:


y = df_combine['FinalStatus']
df_combine.drop(['FinalStatus'],axis=1,inplace=True)

df_combine.head()


# In[76]:


df_combine.insert(29,'FinalStatus',y)


# In[77]:


df_combine


# In[78]:


#considering only numeric columns
numeric_columns= df_combine.select_dtypes(include='number')
numeric_columns


# In[79]:


df_num = df_combine[numeric_columns.columns[1:]]
df_num


# In[80]:


df_combine.columns


# In[81]:


df_num['StageofDisease'].value_counts()


# In[82]:


#To label encode target variable
label_encoder = LabelEncoder()
df_combine['FinalStatus_encoded'] = label_encoder.fit_transform(df_combine['FinalStatus'])
df_combine


# In[83]:


oiginal_target = df_combine['FinalStatus']
df_combine.drop(['FinalStatus'],axis=1,inplace=True)
print(df_combine.shape)
df_combine.head()


# ##### b. Conduct an ANOVA test to determine if there are significant differences in ‘Prothrombin’ levels among different ‘Stages’ of the disease, explain your analysis.

# In[84]:


df2 = df_combine[['StageofDisease', 'ProthrombinLevel']]

# Group data by StageOfDisease
groups = df2.groupby('StageofDisease')['ProthrombinLevel'].apply(list)

# Perform one-way ANOVA
f_val, p_val = stats.f_oneway(*groups)

print(f'F-Value: {f_val}, P-Value: {p_val}')


# As we can see the P-value is less than alpha =0.05, hence the anova test is statistically significant.

# In[85]:


# df_combine.groupby(['ProthrombinLevel']).agg(['count','mean','median','std'])['StageofDisease']


# In[86]:


plt.figure(figsize=[12,8])
sns.lineplot(data=df_combine,x='FollowupDays', y='ProthrombinLevel', hue='StageofDisease', marker='o')


# If ProthrombinLevel  are more then there are chances of serious liver damage, or clotting time will be higher.
# Hence ProthrombinLevel varies significantly with stage of the disease.

# ##### C.Is there a significant difference in the average age of patients based on medications given?

# In[87]:


data = df[['Medication', 'Age']]

groups = data.groupby('Medication')['Age'].apply(list)
f_val, p_val = stats.f_oneway(*groups)

print(f'F-Value: {f_val}, P-Value: {p_val}')


# Yes there is significant difference in the average age of patients based on medications gievn

# #### 5.	Modelling – 

# a.	Divide the data for training and testing. 
# 
# b.	Train any classifier model on the training dataset and evaluate it on the testing dataset. 
# 
# c.	Explain your choice of evaluation metric.
# 
# d.	 Provide further steps that can be used to improve model performance.
# 

# In[88]:


df_combine.columns


# In[89]:


df_combine.head(2)


# In[90]:


y= df_combine['FinalStatus_encoded']
y


# In[91]:


X = df_combine.drop(columns=['PatientID', 'FinalStatus_encoded'])
X


# In[92]:


X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=123)


# In[93]:


print(X_train.shape)


# In[94]:


print(X_test.shape)


# In[95]:


label_encoder.inverse_transform([0,1,2])


# we have not scaled the data as we are going for RandomForest and DecisionTree and XGBoost, but for pipeline we are considering both

# In[96]:


#original peipline
rf_original = RandomForestClassifier(random_state=42)
dt_original = DecisionTreeClassifier(random_state=42)
xgb_original = XGBClassifier(random_state=42)
pipeline_rf_original = Pipeline([
    ('classifier', rf_original)
])
pipeline_dt_original = Pipeline([
    ('classifier', dt_original)
])
pipeline_xgb_original = Pipeline([
    ('classifier', xgb_original)
])


# In[97]:


#scaling pipeline
scaler = StandardScaler()
rf_scaled = RandomForestClassifier(random_state=42)
dt_scaled = DecisionTreeClassifier(random_state=42)
xgb_scaled = XGBClassifier(random_state=42)
pipeline_rf_scaled = Pipeline([
    ('scaler', scaler),
    ('classifier', rf_scaled)
])
pipeline_dt_scaled = Pipeline([
    ('scaler', scaler),
    ('classifier', dt_scaled)
])
pipeline_xgb_scaled = Pipeline([
    ('scaler', scaler),
    ('classifier', xgb_scaled)
])


# In[98]:


results =[]

def evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    return cm, acc, precision, recall




#create pipeline for original and scaled data, as scaling is not really required but this is part of experiement
#for original data
for name, pipeline in {'Random Forest (Original)': pipeline_rf_original,
                       'Decision Tree (Original)': pipeline_dt_original,
                       'XGBoost (Original)': pipeline_xgb_original}.items():
    conf_mat, acc, precision, recall = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
    results.append([name, conf_mat, acc, precision, recall])

#for scaled data
for name, pipeline in {'Random Forest (Scaled)': pipeline_rf_scaled,
                       'Decision Tree (Scaled)': pipeline_dt_scaled,
                       'XGBoost (Scaled)': pipeline_xgb_scaled}.items():
    conf_mat_scale, acc_scale, precision_scale, recall_scale = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
    results.append([name, conf_mat_scale, acc_scale, precision_scale, recall_scale ])


# In[99]:


results


# In[100]:


pd.DataFrame(results)


# In[101]:


results_df = pd.DataFrame(results, columns=['Model', 'Confusion Matrix', 'Accuracy', 'Precision', 'Recall'])

print(results_df)


# In[102]:


results_df


# RandomForest is giving us the best results with max accuracy , recall and precision

# In[103]:


get_ipython().run_cell_magic('time', '', 'n_estimators = [25,50,100]\nmax_depth = [15,20]\nmin_samples_leaf = [2,5,7]\nbootstrap = [True, False]\n\nparam_grid = {\n    "n_estimators": n_estimators,\n    "max_depth": max_depth,\n    "min_samples_leaf": min_samples_leaf,\n    "bootstrap": bootstrap,\n}\n\nrf = RandomForestClassifier(random_state=42)\n\nrf_model = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, verbose=10, n_jobs=-1)\nrf_model.fit(X_train, y_train)\n\nprint("Using hyperparameters ", rf_model.best_params_)\n')


# In[104]:


#using above parameters
rf = RandomForestClassifier(bootstrap= False, max_depth=20, min_samples_leaf= 2, n_estimators= 50,random_state=42)


# In[105]:


rf.fit(X_train,y_train)


# In[106]:


y_pred = rf.predict(X_test)


# In[107]:


confusion_matrix(y_test,y_pred)


# In[108]:


print(classification_report(y_test,y_pred))


# #### overall we are getting good accuracy on test data which is around 92%

# #### If we focus on the recall value for each class then out of all the censored data,91% were correctly predicted, censored due to transplant 100% predicted correctl and from death data 82% were predicted correctly

# we also tried to improve the performance of the model by hyperparameter tuning.

# In[ ]:


# 'censored', 'censored due to transplant', 'death' 0,1,2 


# #### 4.	What are the important features if you need to predict the final status? Please explain the solution using charts.

# #### 6.Which features contribute the most to the progression of the disease?

# In[109]:


feature_importance = rf.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
plt.title('Feature Importance')


# In[110]:


X_test.columns


# In[111]:


feature_imp_df = pd.DataFrame({'Feature':X_test.columns.tolist(),'Feature_importance':feature_importance})
feature_imp_df.sort_values(by='Feature_importance',ascending=False,inplace=True)
feature_imp_df.reset_index(drop=True,inplace=True)
feature_imp_df


# In[112]:


feature_importance


# In[113]:


df.columns


# ##### Based on the graph above "Age","FollowupDays", "BilirubinLevel", "PlateletsCount",'ProthrombinLevel' contributes significantly to the progression of the disease, which are also important in predicting the FinalStatus Column

# #### 7.Use unsupervised methods to see if you can find any insights from the dataset. Provide your insights with facts/charts.	

# In[114]:


for col in df_combine:
    print(col)
    sns.scatterplot(x=col, y='FinalStatus_encoded', data=df_combine)
    plt.show()


# In[115]:


from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 

# Elbow Method to Determine Number of Clusters
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[116]:


X.head()


# In[117]:


n_clusters = 4
kmeans = KMeans(n_clusters=4, random_state=0)

kmeans.fit(X)

pred_y = kmeans.predict(X)



plt.figure(figsize=(12, 6))


for cluster_ in range(n_clusters):
    plt.scatter(X.iloc[pred_y == cluster_, 0], 
                X.iloc[pred_y == cluster_, 1], 
                s=50, cmap='viridis', label=f'Cluster {cluster_}')
# plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=pred_y, s=50, cmap='viridis', label=f'Clusters {n_clusters}')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('K- Means Clustering for Followup and Age')
plt.xlabel('FollowUp')
plt.ylabel('Age')
plt.legend()
plt.show()


# Clusters are formed in such a way that there is a range defined based on the Age column, hence Age wise the Final Status changes

# In[118]:


plt.figure(figsize=(8, 6))
plt.scatter(X.iloc[:, 1], X.iloc[:, 9], c=pred_y, s=50, cmap='viridis', label=f'Clusters {n_clusters}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('K- Means Clustering for Age & PlatelateCount')
plt.xlabel('FollowUp')
plt.ylabel('Age')
plt.legend()
plt.show()


# In[ ]:




