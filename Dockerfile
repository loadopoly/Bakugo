# syntax=docker/dockerfile:1
# Bakugo (cardcenter) — structured-image card scanner with Tesseract OCR.

FROM python:3.12-slim

# tesseract-ocr powers collector-number reading; the two lib packages are the
# minimal native set opencv-python-headless needs on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    CARDCENTER_DB=/data/cardcenter.db

# opencv-python-headless replaces the pyproject's opencv-python (GUI build);
# install deps manually, then the package itself with --no-deps.
RUN pip install --no-cache-dir "numpy>=1.24" "opencv-python-headless>=4.8"

COPY pyproject.toml README.md ./
COPY cardcenter/ ./cardcenter/
RUN pip install --no-cache-dir --no-deps .

RUN mkdir -p /data
VOLUME /data

EXPOSE 8765
HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=5 \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/holders',timeout=4).status==200 else 1)"

CMD ["cardcenter", "--serve", "--lan", "--port", "8765"]
