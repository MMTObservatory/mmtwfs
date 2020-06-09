FROM python:slim

MAINTAINER T. E. Pickering "te.pickering@gmail.com"

RUN apt update && apt -y install tcl

COPY . .

RUN pip install -e .[all,test]
