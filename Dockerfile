# Use python base image
FROM python:3.12-slim


WORKDIR /app

# copy the code into current working directory
COPY . /app/


# Install dependencies for postgres sql
RUN pip install --upgrade pip

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

RUN mkdir -p /app/staticfiles && chown -R 1000:1000 /app/staticfiles

COPY . .

# Set proper permissions
RUN chmod -R 755 /app/staticfiles

# Expose PORT 8000
EXPOSE 8000