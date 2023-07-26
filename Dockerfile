# Use the official Python base image with Python 3.9
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the package files into the container
COPY manic/ /app/manic/
COPY setup.py /app/
COPY README.md /app/
COPY LICENSE /app/

# Install pip dependencies
RUN pip3 install .

# Run the package
CMD ["python3", "-c", "from manic import Manic; print(Manic)"]
