# Data Analysis & Programming (2026-1)

Academic project for the Data Analysis & Programming course. It implements a Medallion-style data architecture orchestrated with Apache Airflow to move data from MongoDB into analytics-ready layers.

## Goal

Build a reproducible data pipeline with two active stages:

- Raw data ingestion into Bronze.
- Data cleaning and standardization into Silver (Parquet format).

## Repository Structure

```text
.
├── airflow/
│   ├── dags/
│   │   ├── bronze_ingestion_dag.py
│   │   └── silver_processing_dag.py
│   ├── docker-compose.yaml
│   ├── Dockerfile
│   ├── requirements.txt
│   └── config/
├── datalake_bronze/
├── datalake_silver/
├── datalake_gold/
├── notebooks/
├── workshop_1/
├── woekshop_2/
├── catchUp/
└── getRawDataService/
```

## Data Architecture

- Bronze (datalake_bronze): raw JSON files.
- Silver (datalake_silver): cleaned and typed Parquet files.
- Gold (datalake_gold): reserved for curated aggregates and final consumption.

## Current DAGs

1. bronze_ingestion_webscraping
- Schedule: daily.
- Action: extracts the most recent document from the web scraping collection and writes it to Bronze.

2. bronze_ingestion_twitter
- Schedule: Monday and Thursday at 06:00 UTC.
- Action: extracts Twitter/comments documents and writes them to Bronze.

3. silver_processing_dag
- Schedule: Monday and Thursday at 07:00 UTC.
- Action: detects new Bronze JSON files, applies cleaning/transformation logic, and writes Parquet files to Silver.

## Requirements

- Docker and Docker Compose.
- Git.
- Optional for local development outside containers: Python 3.10+.

## Environment Configuration

The pipeline uses environment variables for MongoDB connectivity and data lake paths.

Define at least the following in airflow/.env:

```env
MONGO_URI=<your_mongodb_uri>
MONGO_DB=<your_database_name>
BRONZE_BASE_PATH=/opt/airflow/datalake_bronze
SILVER_BASE_PATH=/opt/airflow/datalake_silver
```

Security note:
- Never commit real credentials to GitHub.
- Use placeholders in documentation and keep secrets only in local .env files or a secret manager.

## Quick Start (Airflow with Docker)

From the airflow directory:

```bash
cd airflow
docker compose up airflow-init
docker compose up -d
```

Then open Airflow at http://localhost:8080.

To stop services:

```bash
docker compose down
```

## Expected Data Flow

1. Ingestion DAGs write JSON files into datalake_bronze.
2. The Silver DAG reads the latest file per source.
3. Cleaning and type normalization are applied.
4. Processed Parquet files are written into datalake_silver.

## Workshops and Supporting Material

- workshop_1: first workshop files and deliverables.
- woekshop_2: second workshop technical documentation.
- notebooks: exploratory analysis notebooks.

## Project Status

- Bronze layer: operational.
- Silver layer: operational.
- Gold layer: created, pending modeling and business views.

## Authors

- Juan Diego Grajales Castillo
- Carmen Sofia Florez Juajibioy
- Edgar Alejandro Mora Chala

---

Last updated: 2026-04-12
