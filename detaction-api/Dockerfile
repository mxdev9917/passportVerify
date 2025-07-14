# Use Python 3.10 base image
FROM python:3.10-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxrender1 \
    libxext6 \
    libjpeg-dev \
    zlib1g-dev \
    libleptonica-dev \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libboost-all-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Verify Tesseract installation and find the actual tessdata location
RUN mkdir -p ${TESSDATA_PREFIX} && \
    if [ ! -f "${TESSDATA_PREFIX}/eng.traineddata" ]; then \
        find / -name "eng.traineddata" -exec cp {} ${TESSDATA_PREFIX}/ \; || \
        (echo "Could not find eng.traineddata, downloading..." && \
         wget -O ${TESSDATA_PREFIX}/eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata); \
    fi

# First copy only requirements.txt to cache the pip install step
COPY requirements.txt .

# Install Python packages in two steps to reduce memory pressure
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --no-warn-script-location

# Now copy the rest of the application
COPY . .

# Expose the port your Flask app uses
EXPOSE 5000

# Run your Flask app
CMD ["python", "app.py"]