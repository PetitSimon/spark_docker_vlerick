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
df.rename(columns = {'actor_2_facebook_likes;':'actor_2_facebook_likes'}, inplace = True)
df['actor_2_facebook_likes'] = df['actor_2_facebook_likes'].replace(';','', regex=True)
df['actor_2_facebook_likes'].replace('', np.nan, inplace = True) 
df.actor_2_facebook_likes = df.actor_2_facebook_likes.astype(float, errors = 'ignore')


# In[7]:


df1 = pd.read_csv("../Data/after_release.csv")
df1.head()


# In[8]:


df1.shape


# In[9]:


df1.info()


# In[10]:


df1.describe().round(2)


# The target variable imdb_score column is merged to the pre-sale dataset.

# In[11]:


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


# In[15]:


df[df.language == 'Dutch']


# Looking into the country of the movies in the dataset and 73% of the movies are made in the USA.

# In[16]:


df['country'].value_counts()


# In[17]:


df[df.country == 'Belgium']


# We must be proud of the only Belgian movie that is in the dataset!

# In[18]:


country_count = df['country'].value_counts()
plt.pie(country_count.values, labels=country_count.index,
       autopct='%.0f%%');


# In[19]:


plt.figure(figsize=(10,8))

movie20_df= df_copy.sort_values(by ='imdb_score' , ascending=False)
movie_df_new=movie20_df.head(20)
ax=sns.pointplot(movie_df_new['director_name'], movie_df_new['imdb_score'], hue=movie_df_new['movie_title'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# When looking at the 20 best rated movies I observe that I don't recognize a movie that is in the imdb top 20 on the website. This means that a lot of data is missing compared to the official IMDB list.

# # Data preparation

# Overview of the steps to prepare the data:
# - Missing Values
# - Duplicates
# - Outliers

# ### Missing values

# #### Pre release dataframe

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


# ### Duplicate rows

# #### Pre release dataframe

# duplicate rows for the pre release dataframe

# In[25]:


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
sns.boxplot(data= df['duration']);
print("We observe no outliers in the duration of the movies")


# In[28]:


plt.xticks(rotation=90)

sns.boxplot(data= df[['director_facebook_likes' ,
                     'actor_1_facebook_likes',
                     'actor_2_facebook_likes',
                     'actor_3_facebook_likes',
                     ]]);
print("We observe no outliers in the Facebook likes")


# In[29]:


sns.boxplot(data= df['cast_total_facebook_likes']);
print("We observe no outliers in the Facebook likes")


# In[30]:


sns.boxplot(data= df['budget']);
print("We observe no outliers in the budget variable")


# ## Preprocessing the data

# Dummification of the "genres" variable

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


# In[42]:


# most correlated features
import seaborn as sns
corrmat = df.corr()
plt.figure(figsize = (25,7))
top_corr_features = corrmat.index[abs(corrmat["imdb_score"])>0.1]
g = sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In this colleration matrix, we observe the following:
# There is high multicollinearity among the numerical variables associated with facebook likes. 
# First, it can be observed that there is a logical relationship between actors' facebook likes and the entire cast. 
# Second, there is a high correlation between the facebook likes of the different actors. 
# This would then be a consequence that well-known actors are much more collaborative and less well-known actors are also more collaborative and vice versa. This could be a logical result as the cast of a successful film all get more facebook likes afterwards. In the back of your mind, it should be remembered that the films were all released in the past tense and the facebook likes is a snapshot of now.

# To remove the multicollinearity a new row will replace cast_total_facebook_likes. A new column is made with the sum of the second and thirth actor of the movie.

# In[43]:


df['other_actor_facebook_likes'] = df['actor_3_facebook_likes'] + df['actor_2_facebook_likes']


# In[44]:


df.drop('actor_2_facebook_likes',axis=1,inplace=True)
df.drop('actor_3_facebook_likes',axis=1,inplace=True)
df.drop('cast_total_facebook_likes',axis=1,inplace=True)


# ## Modeling

# ### Linear Regression Analysis

# In[45]:


#Simple Linear Regression => t_Price = B0 + B1t_BHP + e
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True) 
model.fit(df[['budget','director_facebook_likes']], df[['imdb_score']])
print("Intercept or B0: ",model.intercept_)
print("B1: ",model.coef_[0,0])
print("B2: ",model.coef_[0,1])


# In[46]:


#Simple regression model 1
import statsmodels.api as sm
from statsmodels.formula.api import ols
model1 = ols('imdb_score ~ budget + director_facebook_likes + actor_1_facebook_likes + other_actor_facebook_likes', data=df).fit()
print(model1.summary())


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

#######END---ML##########

#step 4
result = spart.dataframes(val_pred)
result.d