FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY . /server
WORKDIR /server
RUN pip install -r requirements.txt
CMD [ "/bin/bash", "-c", "pylint -j 0 $(git ls-files '*.py') --reports=y --errors-only --fail-on=E"]