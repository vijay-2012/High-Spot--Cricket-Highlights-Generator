# Pull tensorflow image with Python3
FROM tensorflow/tensorflow:2.1.0-py3

# Set the working directory to /app
WORKDIR /app

# Transfer content from current dir to /app in container
ADD . /app

# Install python packages
RUN pip install -r requirements.txt

# Start uWSGI using config file
CMD ["uwsgi", "app.ini"]
