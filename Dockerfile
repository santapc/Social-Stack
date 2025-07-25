# Use official Python image
FROM python:3.10-slim

# System dependencies for audio, OCR, PDF
RUN apt-get update && \
    apt-get install -y ffmpeg tesseract-ocr poppler-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Streamlit config: allow external connections
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXsrfProtection=false

# Run the app
CMD ["streamlit", "run", "app.py"] 

#docker build -t mosje-app .
#docker run -p 8501:8501 mosje-app