# Use official tensorflow image:
FROM tensorflow/tensorflow:1.4.1-gpu

# Set working directory:
WORKDIR /home/bertsum

# Install system dependencies:
RUN apt-get clean && apt-get update -y
RUN apt-get install locales -y

# Set locale:
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install python dependencies:
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Cleanup:
RUN rm requirements.txt
