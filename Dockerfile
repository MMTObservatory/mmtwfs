FROM python:3.9

LABEL maintainer="te.pickering@gmail.com"

COPY . .

RUN pip install -e .[all,test]
