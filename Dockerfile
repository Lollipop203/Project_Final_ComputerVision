FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV (Required even for headless if dependencies pull standard cv2)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]
