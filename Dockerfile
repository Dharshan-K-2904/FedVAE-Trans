FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml setup.py README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -e .

CMD ["python", "-m", "demo"]
