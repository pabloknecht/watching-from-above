#FROM python:3.10-slim
FROM tensorflow/tensorflow:2.11.0

COPY wfa /wfa
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py

RUN pip install --upgrade pip
RUN pip install .

# Copy .env with DATA_SOURCE=local and MODEL_TARGET=mlflow
COPY .env .env

CMD uvicorn wfa.api.fast:app --host 0.0.0.0 --reload --port $PORT
