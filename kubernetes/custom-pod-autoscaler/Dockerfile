FROM python:3.9-alpine


# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip3 install kubernetes

# copy project
COPY . /app/

CMD ["python3", "/app/cpa.py"]
#CMD ["bin", "sh"]