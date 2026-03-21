# Data Analysis & Programming - 2026-1

Repository for the **Data Analysis & Programming** course for the 2026-1 semester. This project implements a modern data lake architecture with Apache Airflow for orchestrating data analysis processes.

## 📋 Project Structure

```
.
├── airflow/                    # Data process orchestration with Apache Airflow
│   └── dags/                   # DAGs (Directed Acyclic Graphs) for pipelines
├── datalake_bronze/            # Bronze layer - raw unprocessed data
├── datalake_silver/            # Silver layer - validated and transformed data
├── datalake_gold/              # Gold layer - business-ready data
├── notebooks/                  # Jupyter notebooks for exploratory analysis
└── workshop_1/                 # First workshop - Web Scraping & API Requests
    ├── FirstRequestXApi.json   # Data obtained from API request
    ├── webScraping.txt         # Web scraping notes and exercises
    └── main.tex                # LaTeX document with assignment/notes
```

## 🏗️ Medallion Architecture

The project implements the **Medallion Data Lake** architecture with three data layers:

| Layer | Folder | Description |
|-------|--------|-------------|
| **Bronze** | `datalake_bronze/` | Raw data imported from external sources (APIs, scraping, files) |
| **Silver** | `datalake_silver/` | Clean, validated data with basic transformations |
| **Gold** | `datalake_gold/` | Aggregated and optimized data for analysis and reports |

## 🔄 Main Components

### Apache Airflow (`airflow/`)
- Data pipeline orchestration
- Task dependency management
- Automated monitoring and scheduling
- Reusable DAGs for ETL processes

### Analysis & Notebooks (`notebooks/`)
- Jupyter notebooks for exploratory analysis
- Data visualizations
- Findings documentation

### Workshops
- **Workshop 1**: Introduction to Web Scraping and API Requests

## 🎯 Course Objectives

- Learn data analysis fundamentals
- Implement ETL/ELT pipelines
- Orchestrate processes with Airflow
- Practice web scraping and APIs
- Work with modern data lake architectures

## 📚 Prerequisites

- Python 3.8+
- Apache Airflow
- Jupyter Notebook
- Libraries: pandas, requests, beautifulsoup4, etc.

## 🚀 Getting Started

1. **Clone the repository**
```bash
git clone <repository-url>
cd Data-Analysis-Programming
```

2. **Install dependencies** (optional)
```bash
pip install -r requirements.txt
```

3. **Explore the workshops**
   - Review documentation in `workshop_1/`
   - Run analysis notebooks

4. **Set up Airflow** (optional)
```bash
airflow db init
airflow scheduler
airflow webserver
```

## 📝 Workshop Content

### Workshop 1: Web Scraping & API Requests
- Practical web scraping exercises
- REST API consumption
- JSON data processing

## 👤 Author

Student of Data Analysis & Programming - 2026-1

## 📅 Academic Period

Semester 2026-1

---

*Last updated: March 2026*
