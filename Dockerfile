FROM python:3.10-slim
WORKDIR /review-predictionkyapbwefxcmy
STOPSIGNAL SIGINT

ENV LISTEN_PORT 8000

# System dependencies
RUN apt update && apt install -y libgomp1
RUN pip3 install poetry

COPY . .

RUN poetry config virtualenvs.create false
RUN poetry config installer.parallel false
RUN poetry install --no-interaction --no-ansi --no-dev

ENTRYPOINT uvicorn review-predictionkyapbwefxcmy.serving.serve:app --host 0.0.0.0 --port $LISTEN_PORT --workers 2