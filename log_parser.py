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

from pyspark.sql.functions import col, count, window, to_timestamp, concat, lit

# 1. Combine Date and Time into a proper Timestamp type
# Loghub format: Date '081109', Time '203518' -> 'yyMMdd HHmmss'
parsed_logs = clean_logs.withColumn(
    "Timestamp", 
    to_timestamp(concat(col("Date"), lit(" "), col("Time")), "yyMMdd HHmmss")
)

# 2. Create Behavioral Profiles: Count events per Component and Level in 1-minute windows
feature_matrix = parsed_logs.groupBy(
    window(col("Timestamp"), "1 minute"),
    col("Component"),
    col("Level")
).agg(
    count("*").alias("EventFrequency")
).orderBy("window")

print("\n--- Behavioral Feature Matrix (1-Minute Windows) ---")
feature_matrix.show(10, truncate=False)

# 3. Anomaly Detection via K-Means Clustering
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer

# Convert text strings (Component/Level) into numerical indices
indexer_comp = StringIndexer(inputCol="Component", outputCol="Component_Index")
indexer_level = StringIndexer(inputCol="Level", outputCol="Level_Index")
indexed_logs = indexer_comp.fit(feature_matrix).transform(feature_matrix)
indexed_logs = indexer_level.fit(indexed_logs).transform(indexed_logs)

# Combine numerical columns into a single ML Vector
assembler = VectorAssembler(
    inputCols=["Component_Index", "Level_Index", "EventFrequency"], 
    outputCol="raw_features"
)
vectorized_data = assembler.transform(indexed_logs)

# Scale the features so massive frequencies don't overpower the component types
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
scaled_data = scaler.fit(vectorized_data).transform(vectorized_data)
# Train K-Means to find the 3 most common behavioral baselines (k=3)
kmeans = KMeans(featuresCol="features", predictionCol="Cluster_Assignment", k=3, seed=42)
model = kmeans.fit(scaled_data)
predictions = model.transform(scaled_data)

print("\n--- MLlib Predictions: Normal Clusters ---")
predictions.select("window", "Component", "Level", "EventFrequency", "Cluster_Assignment").show(10, truncate=False)

from pyspark.sql.functions import col, when, sqrt, pow, lit

# Extract centers from the model
c0 = model.clusterCenters()[0]
c1 = model.clusterCenters()[1]
c2 = model.clusterCenters()[2]

# Hardcode Euclidean distance using the ORIGINAL columns we fed into the VectorAssembler
dist0 = sqrt(pow(col("Component_Index") - lit(float(c0[0])), 2) + pow(col("Level_Index") - lit(float(c0[1])), 2) + pow(col("EventFrequency") - lit(float(c0[2])), 2))
dist1 = sqrt(pow(col("Component_Index") - lit(float(c1[0])), 2) + pow(col("Level_Index") - lit(float(c1[1])), 2) + pow(col("EventFrequency") - lit(float(c1[2])), 2))
dist2 = sqrt(pow(col("Component_Index") - lit(float(c2[0])), 2) + pow(col("Level_Index") - lit(float(c2[1])), 2) + pow(col("EventFrequency") - lit(float(c2[2])), 2))

# Assign the correct distance based on which cluster the log was assigned to
anomalies = predictions.withColumn("Anomaly_Score", 
    when(col("Cluster_Assignment") == 0, dist0)
    .when(col("Cluster_Assignment") == 1, dist1)
    .otherwise(dist2)
)

# Extract the absolute weirdest system behaviors (furthest from normal baselines)
top_anomalies = anomalies.orderBy(col("Anomaly_Score").desc())

print("\n--- TOP 20 CRITICAL BEHAVIORAL ANOMALIES ---")
top_anomalies.select("window", "Component", "Level", "EventFrequency", "Anomaly_Score").show(20, truncate=False)

# 6. Save results for the UI Dashboard
results = top_anomalies.limit(20).collect()
with open('/home/hduser/anomalies.csv', 'w') as f:
    f.write("Window,Component,Level,EventFrequency,AnomalyScore\n")
    for row in results:
        f.write(f"{row['window']},{row['Component']},{row['Level']},{row['EventFrequency']},{row['Anomaly_Score']}\n")

spark.stop()
