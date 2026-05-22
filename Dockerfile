FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir kubernetes

COPY lane_scheduler/ lane_scheduler/
RUN pip install --no-cache-dir --no-deps -e .

USER nobody

CMD ["lane-scheduler"]
