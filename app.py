from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
import logging
from logging.handlers import RotatingFileHandler
import time
import uuid


logger = logging.getLogger("model_api")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

fh = RotatingFileHandler("app.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
fh.setLevel(logging.INFO)

fmt = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
ch.setFormatter(fmt)
fh.setFormatter(fmt)

if not logger.handlers:
    logger.addHandler(ch)
    logger.addHandler(fh)

class InputData(BaseModel):
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float

scaler = joblib.load("Scaler.pkl")
model = joblib.load("model.pkl")
MODEL_NAME = "loan-default-model"
MODEL_VERSION = "1.0.0"

app = FastAPI(title="Model API", version="1.0")

@app.middleware("http")
async def add_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    logger.info(
        f"request_start | request_id={request_id} | method={request.method} | path={request.url.path} | client={request.client.host}"
    )

    try:
        response = await call_next(request)
        status = response.status_code
    except Exception as exc:
        status = 500
        logger.exception(f"request_error | request_id={request_id} | path={request.url.path} | error={exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "request_id": request_id},
        )

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"request_end | request_id={request_id} | status={status} | duration_ms={duration_ms:.2f}"
    )

    response.headers["X-Request-ID"] = request_id
    return response

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "version": MODEL_VERSION}

@app.post("/predict/")
def predict(input_data: InputData, request: Request):
    start = time.perf_counter()

    x_values = np.array([[input_data.x1, input_data.x2, input_data.x3, input_data.x4, input_data.x5]])

    scaled_x_values = scaler.transform(x_values)
    prediction = model.predict(scaled_x_values)
    prediction = int(prediction[0])

    duration_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "inference | model=%s | version=%s | prediction=%s | latency_ms=%.2f",
        MODEL_NAME, MODEL_VERSION, prediction, duration_ms
    )

    return {"prediction": prediction, "latency_ms": round(duration_ms, 2), "model": MODEL_NAME, "version": MODEL_VERSION}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
