FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

ENV TOKENIZERS_PARALLELISM=false
ENV POETRY_VERSION=2.1.3

# install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential git \
      libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install "poetry==${POETRY_VERSION}"

# Copy project files
COPY pyproject.toml poetry.lock* ./app/
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --with-GPU

COPY . /app

CMD ["bash", "-c", "poetry run tensorboard --logdir=outputs --port=6006 & poetry run python -m models.train"]
