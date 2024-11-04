# Use the official Python image from DockerHub
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the local application code to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 8080

# Set the environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production



# Dockerfile update: Change CMD to use gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
