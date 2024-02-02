# syntax=docker/dockerfile:1.2
FROM python:latest
# put you docker configuration here
RUN mkdir .pip
COPY requirements.txt /tmp/
RUN --mount=type=secret,id=pip-secret,dst=/root/.pip/pip.conf \
    pip install --upgrade pip && pip install -r /tmp/requirements.txt

ARG product_version
RUN --mount=type=secret,id=pip-secret,dst=/root/.pip/pip.conf \
    pip install delay-model-manu==$product_version

COPY . /app/
WORKDIR /app

EXPOSE 8080

CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:8080", "--timeout=0","--worker-class=uvicorn.workers.UvicornWorker", "delay-model-manu"]