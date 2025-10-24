import os
import subprocess
import sys
import time
import logging
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    year, month, dayofmonth,
    avg, count, max as spark_max, min as spark_min, sum, trim,
    expr, ceil, percentile_approx, lit,
    input_file_name, try_to_timestamp, to_timestamp,
    regexp_extract, col, from_unixtime, hour, desc,
    rand, concat_ws
)

# Setup Spark Session
spark = SparkSession.builder \
    .appName("Container_Local") \
    .getOrCreate()

df = (
    spark.read.option("recursiveFileLookup", "true").text("data/sample/**/*.log")
    .select(
        regexp_extract('value', r'^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})', 1).alias('timestamp'),
        input_file_name().alias('file_path')
    )
    .filter(col('timestamp').isNotNull())
    .filter(col('timestamp') != '')
    .withColumn('application_id', regexp_extract('file_path', r'application_(\d+_\d+)', 0))
    .withColumn('container_id', regexp_extract('file_path', r'(container_\d+_\d+_\d+_\d+)', 1))
    .withColumn('timestamp', to_timestamp('timestamp', "yy/MM/dd HH:mm:ss"))
    ## Extract last 4 characters from application ID and name it app_num column
    .withColumn('app_num', regexp_extract('application_id', r'_(\d{4})$', 1))
    # Extract cluster ID from container ID
    .withColumn('cluster_id', regexp_extract('container_id', r'container_(\d+)_\d+_\d+_\d+', 1))
)

df.show(5, truncate=False)

# Problem 1: Application start and end times
time_series_df = (
    df.groupBy("cluster_id", "application_id")
      .agg(
          spark_min("timestamp").alias("start_time"),
          spark_max("timestamp").alias("end_time")
      )
      .orderBy("application_id")
)
time_series_df.toPandas().to_csv("problem2_timeline.csv", index=False)

## Problem 2: Aggregated container statistics
agg_stats_df = (
    df.groupBy("cluster_id")
      .agg(
          count("application_id").alias("total_apps"),
          spark_min("timestamp").alias("first_app_start"),
          spark_max("timestamp").alias("last_app_end")
      )
      .orderBy("cluster_id")
)
agg_stats_df.toPandas().to_csv("problem2_cluster_summary.csv", index=False)

## Problem 3: Overall summary statistics
avg_apps_per_container = agg_stats_df.agg(avg('total_apps')).first()[0]

with open("problem2_stats.txt", "w") as f:
    f.write(f"Total unique clusters: {df.select('cluster_id').distinct().count()}\n")
    f.write(f"Total applications: {df.select('application_id').count()}\n")
    f.write(f"Average applications per cluster: {avg_apps_per_container}\n")
    f.write("Most heavily used containers in order:\n")

    top_containers = agg_stats_df.orderBy(desc("total_apps")).collect()
    for row in top_containers:
        f.write(f"  {row['cluster_id']}: {row['total_apps']} applications\n")

## Problem 4: Bar chart of # of applications per cluster (seaborn/matplotlib)
## Requirements: 
# #  - Number of applications per cluster
##   - Value labels displayed on top of each bar
##   - Color-coded by cluster ID
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

app_counts = agg_stats_df.select("cluster_id", "total_apps").toPandas()

plt.figure(figsize=(10, 6))
bars = plt.bar(app_counts['cluster_id'], app_counts['total_apps'], color="skyblue")
plt.xlabel('Cluster ID')
plt.ylabel('Number of Applications')
plt.title('Number of Applications per Cluster')
plt.xticks(rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom')
plt.tight_layout()
plt.savefig("problem2_bar_chart.png")

## Problem 5: Desnity plot (seaborn/matplotlib)
## Requirements:
##   - Shows job duration distribution for the largest cluster (cluster with most applications)
##   - Histogram with KDE overlay
##   - **Log scale** on x-axis to handle skewed duration data
##   - Sample count (n=X) displayed in title
largest_cluster_id = (
    app_counts.sort_values(by='total_apps', ascending=False).iloc[0]['cluster_id']
)

# Filter for logs that belong to the largest cluster
largest_cluster_logs = (
    df.filter(col("cluster_id") == largest_cluster_id)
      .groupBy("timestamp")
      .agg(count("*").alias("row_count"))
      .toPandas()
)

# Plot the distribution of rows over time for the largest cluster

plt.figure(figsize=(10, 6))
sns.histplot(
        data=largest_cluster_logs,
        x="row_count",
        bins=30,
        kde=True,
        fill=True,
        color="royalblue",
        alpha=0.6,
        log_scale=True
    )
plt.xlabel('Number of Rows per Log File')
plt.ylabel('Density')
plt.title(f'Distribution of Rows per Log File for Largest Cluster (n={len(largest_cluster_logs)})')
plt.tight_layout()
plt.savefig("problem2_density_plot.png")
