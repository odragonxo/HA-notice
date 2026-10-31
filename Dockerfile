FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 DATA_DIR=/data PORT=8099 BACKTEST_MODEL_DIR=/data/models
WORKDIR /app
RUN pip install --no-cache-dir pandas numpy yfinance tzdata
COPY . /app
CMD ["python", "server.py"]
