FROM apache/airflow:3.2.0
<<<<<<< HEAD

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

USER root

RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get clean

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

USER airflow
=======
 
USER root

RUN apt-get update \
	&& apt-get install -y --no-install-recommends openjdk-17-jre-headless \
	&& rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

USER airflow

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
>>>>>>> 9461ad98 (workshop3 mvp)
