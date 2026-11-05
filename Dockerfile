FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Solo lo que necesita el servicio: código + datos procesados + artefactos del modelo
COPY bruma ./bruma
COPY data/processed ./data/processed
COPY artifacts ./artifacts

ENV PORT=8077
EXPOSE 8077

HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz',timeout=4)"
CMD ["sh", "-c", "uvicorn bruma.api.app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
