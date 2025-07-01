from pyspark.sql import SparkSession

# 1. Create SparkSession
spark = SparkSession.builder \
    .appName("Crypto Data Processing") \
    .getOrCreate()

# 2. Load the BTC data
btc_df = spark.read.csv("data/raw/btc_data.csv", header=True, inferSchema=True)

# 3. Show first 5 rows
btc_df.show(5)

# 4. Simple transformation: convert timestamp to date
btc_df = btc_df.withColumnRenamed("timestamp", "date")

# 5. Print schema
btc_df.printSchema()

# 6. Save back to processed folder (optional)
btc_df.write.mode("overwrite").csv("data/processed/btc_spark_output.csv", header=True)

print("Spark processing complete!")

spark.stop()
