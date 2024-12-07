FROM python:3.12

LABEL maintainer="te.pickering@gmail.com"

COPY . .

RUN pip install -e .[test]
