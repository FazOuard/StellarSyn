from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging
import time
import traceback
from typing import List

from helpers import predict, predict_batch
from services.model_loader import load_model_and_tokenizer
from config import APP_NAME, APP_VERSION, DEBUG, LOG_LEVEL, MAX_SEQUENCE_LENGTH

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

# Initialize FastAPI app
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="A text classification API for AI content detection",
    debug=DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")

class BatchInputData(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=10000, description="List of texts to classify (max 10000 texts)")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Binary prediction (0 or 1)")
    probability: float = Field(..., description="Prediction probability")
    text_length: int = Field(..., description="Length of input text")

class BatchPredictionItem(BaseModel):
    prediction: int = Field(..., description="Binary prediction (0 or 1)")
    probability: float = Field(..., description="Prediction probability")
    text_length: int = Field(..., description="Length of input text")

class BatchPredictionResponse(BaseModel):
    predictions: List[BatchPredictionItem] = Field(..., description="List of predictions for each input text")
    total_texts: int = Field(..., description="Total number of texts processed")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    version: str

class ModelInfoResponse(BaseModel):
    name: str
    version: str
    max_sequence_length: int
    description: str

# Startup event
@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    try:
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer()
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.4f}s"
    )

    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc) if DEBUG else "An error occurred"}
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if model is not None else "error",
        model_loaded=model is not None,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        version=APP_VERSION
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    return ModelInfoResponse(
        name="TextClassifier",
        version=APP_VERSION,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        description="AI content classification model"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(data: InputData):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        prediction, probability = predict(
            data.text,
            model,
            tokenizer
        )

        logger.info(f"Prediction made for text of length {len(data.text)}")

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            text_length=len(data.text.split(" "))
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch_endpoint(data: BatchInputData):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not data.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")

    # Validate that all texts are not empty
    for i, text in enumerate(data.texts):
        if not text.strip():
            raise HTTPException(status_code=400, detail=f"Text at index {i} cannot be empty")

    try:
        predictions = predict_batch(
            data.texts,
            model,
            tokenizer
        )

        # Convert to response format
        prediction_items = []
        for i, (prediction, probability) in enumerate(predictions):
            prediction_items.append(BatchPredictionItem(
                prediction=prediction,
                probability=probability,
                text_length=len(data.texts[i].split(" "))
            ))

        logger.info(f"Batch prediction made for {len(data.texts)} texts")

        return BatchPredictionResponse(
            predictions=prediction_items,
            total_texts=len(data.texts)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Batch prediction failed")

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {APP_NAME}",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }
