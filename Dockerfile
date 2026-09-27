FROM quay.io/jupyter/scipy-notebook:latest

WORKDIR /support-ticket-to-kb

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /support-ticket-to-kb
