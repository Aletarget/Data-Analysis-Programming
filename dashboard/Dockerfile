FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DASH_APP=governance_app.py
ENV STORY_APP=storytelling_app.py

CMD ["sh", "-c", "python $DASH_APP", "python $STORY_APP"]