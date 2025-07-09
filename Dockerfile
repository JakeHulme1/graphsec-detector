FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Set up a WORKDIR
WORKDIR /app

# Install python3, pip, venv, git (for submodules)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-venv \
      git \
      nano \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && ln -s /usr/bin/pip3   /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry globally (no cache)
ENV POETRY_VERSION=2.1.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    TOKENIZERS_PARALLELISM=false
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

# Copy in just the metadata & install dependencies
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --with-GPU

# Bring in full project
COPY . .

# Default command: launch TensorBoard on 6006 and then train
CMD ["bash","-lc","poetry run tensorboard --logdir=outputs --port=6006 & poetry run python -m models.train"]