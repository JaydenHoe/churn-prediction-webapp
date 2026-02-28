# Start with a python base image
# Take your pick from https://hub.docker.com/_/python
FROM python:3.12-slim

# Set /flask-app as the main application directory
WORKDIR /flask-app

# Copy the requirements.txt file and required directories into docker image
COPY ./requirements.txt /flask-app/requirements.txt
COPY ./src /flask-app/src
# Use model_artifacts so the "model" Python module (src/model.py) is not shadowed
COPY ./model /flask-app/model_artifacts

# /flask-app so gunicorn finds "src.app"; /flask-app/src so "from model import Model" finds src/model.py
ENV PYTHONPATH=/flask-app:/flask-app/src
ENV CHURN_MODEL_DIR=model_artifacts

# Install python package dependancies, without saving downloaded packages locally
RUN pip install -r /flask-app/requirements.txt --no-cache-dir

# Allow port 80 to be accessed (Flask app)
EXPOSE 80

# Start the Flask app using gunicorn
CMD ["gunicorn", "--bind=0.0.0.0:80", "src.app:app"]