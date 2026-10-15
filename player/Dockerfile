# nist_lite PLAYER image: serves the miner's ONNX policy over gym_v1.
# Vendored gym_v1 (do NOT build FROM apex-player-base). Build context = repo root:
#   docker build -f player/Dockerfile -t nist-assembly-player .
# The policy is NOT baked in: the platform writes it to /app/policy.onnx at run time.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app /data && chown -R 1000:1000 /app /data
WORKDIR /app

COPY player/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY player/gym_v1/ /app/gym_v1/
COPY player/launch.py /app/launch.py

USER 1000
ENV OMP_NUM_THREADS=1
CMD ["python", "/app/launch.py", "--port", "8000"]
