FROM python:3.9-slim

WORKDIR /app

# Only glib is needed for opencv-headless (X11 libs not needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]
