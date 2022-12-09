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

df1 = df_after.toPandas()
df2 = df_pre.toPandas()

