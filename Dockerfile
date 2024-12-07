FROM python:3.12-slim

LABEL maintainer="te.pickering@gmail.com"

COPY . .

RUN pip install -e .[test]
