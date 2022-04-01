FROM python:latest

LABEL maintainer="te.pickering@gmail.com"

COPY . .

RUN pip install -e .[all,test]
