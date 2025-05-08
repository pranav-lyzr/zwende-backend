FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y procps net-tools && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and startup script
COPY . .

# Set permissions and environment variables
RUN chmod +x startup.sh && mkdir -p /app/data
ENV PORT=8003 DATA_DIR=/app/data PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8003

# Run the startup script
CMD ["./startup.sh"]