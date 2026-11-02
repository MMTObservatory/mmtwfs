FROM python:3.15

LABEL maintainer="te.pickering@gmail.com"

COPY . .

RUN pip install -e .[test]
