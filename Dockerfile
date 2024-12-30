FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl build-essential libxml2-dev libxslt-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set up workspace
WORKDIR /app
COPY . /app

# Set Poetry's virtual environment location
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
RUN poetry install --no-dev

# Default command
CMD ["poetry", "run", "python", "-m", "svg_converter"]
