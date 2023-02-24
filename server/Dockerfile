FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
COPY ./requirements.txt /server/requirements.txt
WORKDIR /server
RUN pip install -r requirements.txt
COPY . /server
CMD [ "/bin/bash", "-c", "pylint app/ --reports=y --errors-only --fail-on=E"]