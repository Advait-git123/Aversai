# ---- Base Python Image ----
FROM python:3.11-slim

# ---- Set working directory ----
WORKDIR /app

# ---- Copy project files ----
COPY . .

# ---- Install system dependencies for PyMuPDF (fitz) ----
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python dependencies ----
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Expose API port ----
EXPOSE 8000

# ---- Run FastAPI app ----
CMD ["uvicorn", "Backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

