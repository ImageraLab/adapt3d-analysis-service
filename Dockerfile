FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# Combine these commands for better Docker layer caching and simpler execution
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY main.py .

# Expose port (already there, just confirming)
EXPOSE 8000

# Run the application using the Docker Command we set in Render
# This CMD is secondary to the Render "Docker Command" setting, but good to have as a fallback
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
