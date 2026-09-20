# Render / Fly.io / Railway 등 컨테이너 배포용
FROM python:3.13-slim

# APScheduler 가 Asia/Seoul 을 해석하려면 tzdata 가 필요하다
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Seoul \
    SCHEDULER_TZ=Asia/Seoul \
    ENABLE_SCHEDULER=1 \
    OPT_WORKERS=2 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# DB 를 볼륨에 마운트하면 재시작해도 유지된다 (미마운트 시 컨테이너 내부에 생성)
VOLUME ["/var/data"]

EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\",\"8000\")}/api/status',timeout=8)"

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
