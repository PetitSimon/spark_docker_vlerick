#step 1: Read the CSV data from this S3 bucket using PySpark
from pyspark import SparkConf
from pyspark.sql import SparkSession

BUCKET = "dmacademy-course-assets"
KEY_after = "vlerick/after_release.csv "
KEY_pre = "vlerick/pre_release.csv"

config = {
    "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1",
    "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
}
conf = SparkConf().setAll(config.items())
spark = SparkSession.builder.config(conf=conf).getOrCreate()

df_after = spark.read.csv(f"s3a://{BUCKET}/{KEY_after}", header=True)
df_pre = spark.read.csv(f"s3a://{BUCKET}/{KEY_pre}", header=True)

df_after.show()
df_pre.show()

#step 2: Convert the Spark DataFrames to Pandas DataFrames.

from pyspark.sql import SparkSession
import pandas as pd

df = df_after.toPandas()
df1 = df_pre.toPandas()

#######BEGIN--------ML
#step 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import networkx as nx
import itertools
import matplotlib.pyplot as plt


df_original = df
df_name_col = df[['director_name', 'actor_3_name', 'actor_2_name', 'actor_1_name', 'movie_title']]

df = pd.merge(df,df1[['movie_title','imdb_score']], on='movie_title', how='inner')
df_copy = df


# To check the unique values for each of the categorical variables, you can use the <a href="https://pandas.pydata.org/docs/reference/api/pandas.unique.html">`unique`</a> command.

# # Business understanding of the data

# #### Pre release dataframe

# In[12]:


categoricals = df.select_dtypes(include = 'object').columns.tolist()
#for category in categoricals: 
#    print(category, 'has values:', df[category].unique())


# Looking into the language of the movies in the dataset. With this table, it can be observed that 92 per cent of the films in the dataset are in English.

# In[13]:


c = df['language'].value_counts(dropna=False)
p = df['language'].value_counts(dropna=False, normalize=True)
pd.concat([c,p], axis=1, keys=['counts', '%'])


# In[14]:


language_count = df['language'].value_counts()
plt.pie(language_count.values, labels=language_count.index,
       autopct='%.0f%%');



# In[16]:



# We must be proud of the only Belgian movie that is in the dataset!

# In[18]:


country_count = df['country'].value_counts()
plt.pie(country_count.values, labels=country_count.index,
       autopct='%.0f%%');


# In[19]:







# In[20]:


#Check number of missing values per feature
df.isnull().sum()


# In[21]:


#Check all rows containing missing values
df[df.isnull().any(axis = 1)].shape


# In[22]:


#Remove row containing almost only missing values (10 or more missing values)
df = df[df.isnull().sum(axis=1) < 8]
df.shape


# The lines that has more then 8 missing values are removed from the dataset.

# In[23]:


df.isnull().sum()


# In[24]:


df[df.isnull().any(axis = 1)].shape



#duplicated rows
df[df.duplicated() == True]


# In[26]:


#remove duplicates
df.drop_duplicates(inplace = True)
print(df[df.duplicated() == True].shape[0])


# ### Outliers
# 

# #### Pre release dataframe

# In[27]:


#make a separate boxplot for every variable with the same metrics



# In[28]:





# In[29]:


sns.boxplot(data= df['cast_total_facebook_likes']);
print("We observe no outliers in the Facebook likes")


# In[30]:


# In[31]:


dummies = df["genres"].str.get_dummies("|")
#dummies = dummies.astype(bool)


# In[32]:


df = pd.merge(df, dummies, left_index = True, right_index = True, suffixes = None)

#need to drop the a column in the dummy dataframe
df = df.drop(columns = ['Western','genres'], axis=1)
df


# In[33]:


df.info()


# Dummification using OneHotEncoder

# In the following part the names om the actors, directors and movies will be removed because this columns can not be dummified.

# In[34]:


from sklearn.preprocessing import OneHotEncoder


ohe = OneHotEncoder(drop='if_binary')

cat_feat = ohe.fit(df.select_dtypes('object').drop(columns=['director_name', 'actor_3_name', 'actor_2_name', 'actor_1_name', 'movie_title']))


df = pd.concat([
    df.select_dtypes(exclude='object'),
    pd.DataFrame(
        ohe.transform(df.select_dtypes('object').drop(columns=['director_name', 'actor_3_name', 'actor_2_name', 'actor_1_name', 'movie_title'])).toarray(),
        columns=ohe.get_feature_names_out(),
        index=df.index
    )
], axis=1)

df.head(3)


# In[35]:


df[df.isna().any(axis = 1)]


# Tackling the missing values in the dataframe

# In[36]:


df.isnull().sum()


# In[37]:


df.info()


# Filling the missing values in the dataframe. The missing values o

# In[38]:


num = df[["imdb_score", "duration", "budget"]]
num_likes = df[['director_facebook_likes',
                     'actor_1_facebook_likes',
                     'actor_2_facebook_likes',
                     'actor_3_facebook_likes']]

for column in num_likes:
    df[column].fillna(0, inplace = True) 
for column in num: 
    df[column].fillna(df[column].median(), inplace = True) 


# In[39]:


df.head(5)


# ## Further analysis

# Looking into the target variable

# In[40]:


sns.scatterplot(data = df, x = "imdb_score", y = "budget");


# In[41]:


plt.figure(figsize=(12,8))
sns.histplot(num.imdb_score.values, bins=50, kde=False)
plt.xlabel('imdb_score', fontsize=12)
plt.show()





df['other_actor_facebook_likes'] = df['actor_3_facebook_likes'] + df['actor_2_facebook_likes']


# In[44]:


df.drop('actor_2_facebook_likes',axis=1,inplace=True)
df.drop('actor_3_facebook_likes',axis=1,inplace=True)
df.drop('cast_total_facebook_likes',axis=1,inplace=True)


# ## Modeling

# ### Linear Regression Analysis

# In[45]:





# ### Random forest

# In[47]:


from sklearn.model_selection import train_test_split

X = df.drop(columns='imdb_score')
y = df.imdb_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 5)
print('Train set:', X_train.shape)
print('Test set:', X_test.shape)


# In[48]:


#preparing the dataset by adding the imdb rating
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import tree

rfr = RandomForestRegressor(n_estimators = 100, random_state = 32, max_depth=4, min_samples_leaf=5 ,max_leaf_nodes=8)
rfr.fit(X_train, y_train)
rfr_score_train = rfr.score(X_train, y_train)
print("Training score: ",rfr_score_train)
rfr_score_test = rfr.score(X_test, y_test)
print("Testing score: ",rfr_score_test)


# In[49]:


from sklearn.metrics import mean_squared_error
y_pred = rfr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ",mse)


# In[50]:


np.sqrt(mean_squared_error(y_test, y_pred))


# In[51]:


df_rfr = pd.DataFrame(data = rfr.feature_importances_,index = X_train.columns.tolist())

df_rfr = df_rfr[df_rfr.iloc[:,0] > 0].sort_values(by = 0,ascending = False)
fig, ax = plt.subplots(figsize=(20,10))
sns.barplot(y = df_rfr.index, x= df_rfr[0])
plt.xlabel('importance');


# In[52]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

rsquare = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
pd.DataFrame({'eval_criteria': ['r-square','MAE'],'value':[rsquare,mae]})

val_pred = y_pred
#######END---ML##########

#step 4
result = spark.createDataFrame(val_pred)
reult.show()

#step 5

df.write.json("s3://dmacademy-course-assets/vlerick/Simon/")
