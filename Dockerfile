FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir hatch

# Copy only the files needed to set up the Hatch environment
COPY pyproject.toml ./
COPY README.md ./

# Pre-create the Hatch environment and install dependencies
RUN hatch env create

# Copy the rest of the application
COPY . .

