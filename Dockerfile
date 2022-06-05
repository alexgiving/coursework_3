FROM python:3.8-slim-buster
LABEL maintainer="alexgiving@mail.ru"
USER root
WORKDIR /home/
COPY build_files/* build_files/
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY scripts/prod.py .
CMD python3 prod.py