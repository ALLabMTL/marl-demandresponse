FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY . /server
WORKDIR /server
RUN pip install -r requirements.txt
CMD [ "/bin/bash", "-c", "pylint app/ --reports=y --errors-only --fail-on=E"]