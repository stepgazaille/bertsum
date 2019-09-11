# Use official pytorch image
FROM tensorflow/tensorflow:1.4.1-gpu

# Set the working directory to /usum
WORKDIR /home/bertsum

# Copy scripts directory to container WORKDIR
COPY requirements.txt requirements.txt

# Install dependencies:
RUN apt-get update -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Cleanup:
RUN rm requirements.txt
