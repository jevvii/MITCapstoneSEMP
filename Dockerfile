FROM python:3.12-slim

RUN apt-get update \
    && (apt-get install -y --no-install-recommends bash espeak ffmpeg libespeak1 curl \
        || apt-get install -y --no-install-recommends bash ffmpeg curl) \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN chmod +x ./scripts/*.sh

# Allow build-time declaration of the exposed port (default 8000).
ARG PORT=8000
EXPOSE ${PORT}

CMD ["bash", "./scripts/start-backend.sh"]
