import os
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower, when

load_dotenv()

# Secrets from .env
SNOWFLAKE_OPTIONS = {
    "sfURL": "ar29154.us-central1.gcp.snowflakecomputing.com",
    "sfUser": os.getenv("SNOWFLAKE_USER"),
    "sfPassword": os.getenv("SNOWFLAKE_PASSWORD"),
    # Non-sensitive config hardcoded
    "sfDatabase": "UCI_DIABETES",
    "sfSchema": "RAW",
    "sfWarehouse": "COMPUTE_WH",
    "sfRole": "TRANSFORMER",
    "preactions": "USE DATABASE UCI_DIABETES; USE SCHEMA RAW; USE WAREHOUSE COMPUTE_WH;"
}

def create_spark_session():
    return (
        SparkSession.builder
        .appName("uci_diabetes_ingestion")
        .config(
            "spark.jars.packages",
            "net.snowflake:spark-snowflake_2.12:2.12.0-spark_3.4,"
            "net.snowflake:snowflake-jdbc:3.15.0"
        )
        .getOrCreate()
    )

def read_local_csv(spark, path):
    return (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("nullValue", "?")
        .csv(path)
    )

def clean_column_names(df):
    """Standardize column names to snake_case"""
    cleaned = df
    for col_name in df.columns:
        new_name = (
            col_name
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        cleaned = cleaned.withColumnRenamed(col_name, new_name)
    return cleaned

def write_to_snowflake(df, table_name):
    (
        df.write
        .format("net.snowflake.spark.snowflake")
        .options(**SNOWFLAKE_OPTIONS)
        .option("dbtable", table_name)
        .mode("overwrite")
        .save()
    )

def main():
    spark = create_spark_session()

    raw_path = "data/raw/diabetic_data.csv"
    df = read_local_csv(spark, raw_path)

    print(f"Row count: {df.count()}")
    print(f"Columns: {df.columns}")

    df = clean_column_names(df)

    write_to_snowflake(df, "DAT")

    print("Ingestion complete.")
    spark.stop()

if __name__ == "__main__":
    main()