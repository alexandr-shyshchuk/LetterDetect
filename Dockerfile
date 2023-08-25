FROM ubuntu:latest
MAINTAINER Shyshchuk Olexandr <shyshchuko@gmail.com>

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /model/requirements.txt
WORKDIR /model
RUN pip install -r requirements.txt

COPY . /model

CMD ["python3", "test_inference_script.py", "input"]
