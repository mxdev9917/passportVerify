FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libffi-dev \
    libssl-dev \
    python3-dev \
    libbz2-dev \
    liblzma-dev \
    libjpeg-dev \
    zlib1g-dev \
    libsqlite3-dev \
    libfreetype6-dev \
    libpng-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5000

CMD ["python", "app.py"]
