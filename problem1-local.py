import os
import subprocess
import sys
import time
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    year, month, dayofmonth,
    avg, count, max as spark_max, min as spark_min,
    expr, ceil, percentile_approx
)
from pyspark.sql.functions import regexp_extract
from pyspark.sql.functions import col, from_unixtime, hour, desc
from pyspark.sql.functions import rand, concat_ws
import pandas as pd

## Setup Spark Session
spark = SparkSession.builder \
         .appName("Log_Local") \
         .getOrCreate()

## Read log data and parse
df = spark.read.text("data/sample/**/*.log")
df_parsed = df.select(
    regexp_extract('value', r'^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})', 1).alias('timestamp'),
    regexp_extract('value', r'(INFO|WARN|ERROR|DEBUG)', 1).alias('log_level'),
    regexp_extract('value', r'(INFO|WARN|ERROR|DEBUG)\s+([^:]+):', 2).alias('component'),
    col('value').alias('message')
)

df_parsed.head()

## Problem 1: Log level counts (2 columns: log_level, count)
log_level_counts = df_parsed.groupBy("log_level") \
    .agg(count("*").alias("count")) \
    .orderBy(desc("count"))

log_level_counts.toPandas().to_csv("data/output/problem1_counts.csv", index=False)

## Problem 2: Extract 10 random sample log entries with their log levels
sample_logs = df_parsed.orderBy(rand()).limit(10)
sample_df = sample_logs.select("message", "log_level")
sample_df.toPandas().to_csv("data/output/problem1_sample.csv", index=False)


## Problem 3: Summary statistics; write to text file
logs_count = df_parsed.count()
# log count where log level is not null
logs_with_level_count = df_parsed.filter(col("log_level").isNotNull()).count()
# Unique log levels
unique_log_levels = df_parsed.select("log_level").distinct().count()
# Log level distributions (counts per log level)
log_level_distribution = log_level_counts.collect()

# Write to .txt
with open("data/output/problem1_summary.txt", "w") as f:
    f.write(f"Total log lines processed: {logs_count}\n")
    f.write(f"Total lines with log levels: {logs_with_level_count}\n")
    f.write(f"Unique log levels: {unique_log_levels}\n")
    f.write("Log level distribution:\n")
    for row in log_level_distribution:
        f.write(f"  {row['log_level']}: {row['count']}\n")