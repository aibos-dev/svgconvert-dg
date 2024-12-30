# Use a Python base image
FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && apt-get clean

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set Poetry's virtual environment location to be inside the project
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# Set the working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace

# Install project dependencies
RUN poetry install
