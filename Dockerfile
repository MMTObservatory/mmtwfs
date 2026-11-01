FROM python:3.14

LABEL maintainer="te.pickering@gmail.com"

COPY . .

RUN pip install -e .[test]
