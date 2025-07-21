# Use python base image
FROM python:3.13.2-alpine3.20


WORKDIR /app

# copy the code into current working directory
COPY . /app/

# Install dependencies
RUN apk add --no-cache \
    openssl \ 
    build-base \
    python3-dev \
    musl-dev \
    linux-headers \
    ethtool

# Install dependencies for postgres sql
RUN pip install --upgrade pip
RUN apk add --no-cache postgresql-dev

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

RUN mkdir -p /app/staticfiles && chown -R 1000:1000 /app/staticfiles

COPY . .

# Set proper permissions
RUN chmod -R 755 /app/staticfiles

# Expose PORT 8000
EXPOSE 8000