FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY app.py .
COPY vvipvoice_v5_ref.wav /app/vvipvoice_v5_ref.wav
COPY Fangyung_vvipvoice.wav /app/Fangyung_vvipvoice.wav
COPY Htun_vvipvoice.wav /app/Htun_vvipvoice.wav
COPY Pyae_vvipvoice.wav /app/Pyae_vvipvoice.wav
COPY Phyo_vvipvoice.wav /app/Phyo_vvipvoice.wav
CMD ["python","app.py"]
