FROM python:latest

MAINTAINER T. E. Pickering "te.pickering@gmail.com"

COPY . .

RUN pip install scipy=1.3
RUN pip install -e .[all,test]
