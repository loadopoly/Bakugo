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

# Unprivileged runtime user. The UID is fixed so docker-compose's volume-init
# step can chown an existing /data volume to match.
RUN groupadd --system --gid 10001 bakugo \
    && useradd --system --uid 10001 --gid bakugo --home-dir /home/bakugo \
        --create-home --shell /usr/sbin/nologin bakugo

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/bakugo \
    CARDCENTER_DB=/data/cardcenter.db \
    OPENCV_IO_MAX_IMAGE_PIXELS=64000000

# opencv-python-headless replaces the pyproject's opencv-python (GUI build);
# install deps manually, then the package itself with --no-deps.
# duckdb powers the OLAP analytical layer (always available in Docker).
RUN pip install --no-cache-dir "numpy>=1.24" "opencv-python-headless>=4.8" "duckdb>=1.0.0"

COPY pyproject.toml README.md ./
COPY cardcenter/ ./cardcenter/
# Refuse the build if the package tree holds a collection-derived or untagged
# artifact (see cardcenter/release_guard.py). The private trainer tree is not
# copied into this image at all.
RUN python -m cardcenter.release_guard /app/cardcenter
RUN pip install --no-cache-dir --no-deps .

RUN mkdir -p /data/parquet && chown -R bakugo:bakugo /data
VOLUME /data

USER bakugo

# analytics.py runs `INSTALL sqlite` on first use. The container has no
# internet access at runtime, so fetch the extension into $HOME at build time.
RUN python -c "import duckdb; duckdb.connect().execute('INSTALL sqlite; LOAD sqlite;')"

EXPOSE 8765
HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=5 \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=4).status==200 else 1)"

# Listens on all interfaces *inside* the container; docker-compose decides
# what can reach that (only the caddy proxy, over an internal network).
CMD ["cardcenter", "--serve", "--lan", "--port", "8765"]
