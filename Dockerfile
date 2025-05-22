
# Use a slim Python 3.12 base image
FROM python:3.12-slim

# Environment variables for cleaner logging and no .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory inside the container
WORKDIR /app

# Install system dependencies needed for pip + science packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY requirements.txt .

# Upgrade pip and install Python packages
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy all app code into the container
COPY . .

# Expose the port used by FastAPI
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
