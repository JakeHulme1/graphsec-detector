FROM python:3.11-slim

WORKDIR /app

ENV TOKENIZERS_PARALLELISM=false
ENV POETRY_VERSION=2.1.3

# Install poetry
RUN pip install "poetry==${POETRY_VERSION}"

# Copy project files
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --with-GPU

COPY . .

CMD ["bash", "-c", "poetry run tensorboard --logdir=outputs --port=6006 & poetry run python -m models.train"]
