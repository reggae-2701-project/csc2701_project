# Use the official Python base image
FROM python:3.11.13-slim
#FROM python:3.11.14-alpine3.22

#RUN apk add --no-cache \
#   build-base \
#    gcc \
#    gfortran \
#    musl-dev \
#    openblas-dev \
#    lapack-dev \
#    python3-dev \
#    py3-numpy \
#  py3-scipy

# Set the working directory inside the container
WORKDIR /app/backend

# Copy the requirements file to the working directory
COPY backend/requirements.txt .
COPY backend/requirements_h.txt .

# Install the Python dependencies
RUN pip install  --no-cache-dir -r requirements.txt \
 && pip install  --no-cache-dir -r requirements_h.txt

# Copy the application code to the working directory
COPY backend/ .

# Expose the port on which the application will run
EXPOSE 8080

# Run the FastAPI application using uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]