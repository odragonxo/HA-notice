FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 DATA_DIR=/data PORT=8099
WORKDIR /app
RUN pip install --no-cache-dir pandas numpy yfinance tzdata
COPY . /app
CMD ["python", "server.py"]
