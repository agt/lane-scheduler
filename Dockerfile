FROM python:3.11.10-slim

WORKDIR /app

COPY pyproject.toml .
COPY lane_scheduler/ lane_scheduler/
RUN pip install --no-cache-dir -e .

USER nobody

CMD ["lane-scheduler"]
