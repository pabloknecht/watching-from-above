#FROM python:3.10-slim
FROM tensorflow/tensorflow:2.10.0


COPY wfa /wfa
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py

RUN pip install --upgrade pip
RUN pip install .
#RUN pip install setup.py

CMD uvicorn wfa.api.fast:app --host 0.0.0.0 --reload --port $PORT
