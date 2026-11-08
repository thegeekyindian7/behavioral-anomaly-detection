from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract

# 1. Initialize Spark Session connected to YARN
spark = SparkSession.builder \
    .appName("Behavioral_Anomaly_Parser") \
    .getOrCreate()

print("Spark Session successfully initialized.")

# 2. Load the raw text file from HDFS
hdfs_path = "/user/anomaly_project/raw_data/HDFS.log"
raw_logs = spark.read.text(hdfs_path)

# 3. Define Regex pattern for the HDFS log format
regex_pattern = r'^(\d{6})\s+(\d{6})\s+(\d+)\s+(\w+)\s+(.*?):\s+(.*)$'

# 4. Extract fields into a structured DataFrame
parsed_logs = raw_logs.select(
    regexp_extract('value', regex_pattern, 1).alias('Date'),
    regexp_extract('value', regex_pattern, 2).alias('Time'),
    regexp_extract('value', regex_pattern, 3).alias('PID'),
    regexp_extract('value', regex_pattern, 4).alias('Level'),
    regexp_extract('value', regex_pattern, 5).alias('Component'),
    regexp_extract('value', regex_pattern, 6).alias('Message')
)

# 5. Filter out empty rows (logs that didn't match the regex)
clean_logs = parsed_logs.filter(col("Date") != "")

# 6. Show the structured data 
print("--- FIRST 10 PARSED LOGS ---")
clean_logs.show(10, truncate=False)

# 7. Count total rows (Triggers distributed action)
total_count = clean_logs.count()
print(f"Total structured logs processed: {total_count}")

spark.stop()
