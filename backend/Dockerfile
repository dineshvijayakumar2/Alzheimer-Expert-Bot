FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (if any needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy all files from backend directory
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for logs
RUN mkdir -p /app/logs

# Create log files with proper permissions
RUN mkdir -p /app/data /app/logs && \
    touch /app/logs/app.log /app/logs/error.log && \
    chmod -R 777 /app/logs /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODULE_NAME=app
ENV VARIABLE_NAME=app
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

