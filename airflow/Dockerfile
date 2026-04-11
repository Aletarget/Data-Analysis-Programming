FROM apache/airflow:3.2.0
 
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt