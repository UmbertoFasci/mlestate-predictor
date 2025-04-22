FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

# Copy wheels from builder stage
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m appuser
USER appuser

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Expose the port the app runs on
EXPOSE $PORT

# Run the application
CMD gunicorn -w 3 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:$PORT