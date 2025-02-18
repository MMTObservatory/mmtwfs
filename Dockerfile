FROM python:3.13

LABEL maintainer="te.pickering@gmail.com"

COPY . .

RUN pip install -e .[test]
