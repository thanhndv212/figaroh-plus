# syntax=docker/dockerfile:experimental
FROM python:3.8

ADD requirements.txt .

RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,sharing=locked,target=/var/lib/apt \
    --mount=type=cache,sharing=locked,target=/root/.cache \
    apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -qqy \
    coinor-libipopt-dev \
    gcc \
    libblas-dev \
    liblapack-dev \
 && python -m pip install -U pip \
 && python -m pip install -U wheel \
 && python -m pip install -U -r requirements.txt \
 && apt-get autoremove -qqy gcc

WORKDIR /figaroh
ADD . .
CMD python Identification_pipeline_simulated_data_main.py
