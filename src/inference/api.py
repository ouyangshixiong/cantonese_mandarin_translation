"""
FastAPI service for Cantonese-Mandarin translation
"""
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from .translator import CantoneseTranslator

logger = logging.getLogger(__name__)


class TranslationRequest(BaseModel):
    """Translation request model"""
    text: str = Field(..., description="Text to translate")
    source_lang: str = Field(default="cantonese", description="Source language")
    target_lang: str = Field(default="mandarin", description="Target language")
    beam_size: int = Field(default=4, ge=1, le=10, description="Beam search size")


class TranslationResponse(BaseModel):
    """Translation response model"""
    text: str = Field(..., description="Translated text")
    source_lang: str = Field(..., description="Source language")
    target_lang: str = Field(..., description="Target language")
    inference_time: float = Field(..., description="Inference time in seconds")


class BatchTranslationRequest(BaseModel):
    """Batch translation request model"""
    texts: List[str] = Field(..., description="List of texts to translate")
    source_lang: str = Field(default="cantonese", description="Source language")
    target_lang: str = Field(default="mandarin", description="Target language")
    beam_size: int = Field(default=4, ge=1, le=10, description="Beam search size")


class BatchTranslationResponse(BaseModel):
    """Batch translation response model"""
    translations: List[str] = Field(..., description="List of translated texts")
    source_lang: str = Field(..., description="Source language")
    target_lang: str = Field(..., description="Target language")
    total_time: float = Field(..., description="Total processing time in seconds")


class ModelInfoResponse(BaseModel):
    """Model information response model"""
    model_name: str = Field(..., description="Model name")
    framework: str = Field(..., description="Framework used")
    device: str = Field(..., description="Device used")
    total_parameters: int = Field(..., description="Total number of parameters")
    trainable_parameters: int = Field(..., description="Number of trainable parameters")
    max_length: int = Field(..., description="Maximum sequence length")


class TranslationService:
    """Translation service class"""
    
    def __init__(self, model_name: str = "Tencent-Hunyuan/Hunyuan-MT-7B", 
                 framework: str = "pytorch", device: str = "auto"):
        self.translator = CantoneseTranslator(
            model_name=model_name,
            framework=framework,
            device=device
        )
    
    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate a single text"""
        import time
        start_time = time.time()
        
        try:
            result = self.translator.translate(
                text=request.text,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                beam_size=request.beam_size
            )
            
            inference_time = time.time() - start_time
            
            return TranslationResponse(
                text=result,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                inference_time=inference_time
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise
    
    async def batch_translate(self, request: BatchTranslationRequest) -> BatchTranslationResponse:
        """Batch translate multiple texts"""
        import time
        start_time = time.time()
        
        try:
            results = self.translator.batch_translate(
                texts=request.texts,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                beam_size=request.beam_size
            )
            
            total_time = time.time() - start_time
            
            return BatchTranslationResponse(
                translations=results,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                total_time=total_time
            )
            
        except Exception as e:
            logger.error(f"Batch translation failed: {str(e)}")
            raise
    
    def get_model_info(self) -> ModelInfoResponse:
        """Get model information"""
        info = self.translator.get_model_info()
        
        return ModelInfoResponse(
            model_name=info["model_name"],
            framework=info["framework"],
            device=info["device"],
            total_parameters=info["total_parameters"],
            trainable_parameters=info["trainable_parameters"],
            max_length=info["max_length"]
        )


def create_app(service: TranslationService):
    """Create FastAPI application"""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        logger.error("FastAPI not available. Install with: pip install fastapi uvicorn")
        return None
    
    app = FastAPI(
        title="Cantonese-Mandarin Translation API",
        description="API for translating between Cantonese and Mandarin using Hunyuan-MT-7B",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Cantonese-Mandarin Translation API",
            "version": "1.0.0",
            "endpoints": [
                "/translate",
                "/batch_translate",
                "/model_info",
                "/health"
            ]
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "service": "translation"}
    
    @app.post("/translate", response_model=TranslationResponse)
    async def translate(request: TranslationRequest):
        """Translate a single text"""
        try:
            return await service.translate(request)
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/batch_translate", response_model=BatchTranslationResponse)
    async def batch_translate(request: BatchTranslationRequest):
        """Batch translate multiple texts"""
        try:
            return await service.batch_translate(request)
        except Exception as e:
            logger.error(f"Batch translation error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/model_info", response_model=ModelInfoResponse)
    async def model_info():
        """Get model information"""
        try:
            return service.get_model_info()
        except Exception as e:
            logger.error(f"Model info error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def run_api_server(model_name: str = "Tencent-Hunyuan/Hunyuan-MT-7B",
                  framework: str = "pytorch",
                  device: str = "auto",
                  host: str = "0.0.0.0",
                  port: int = 8000):
    """Run the API server"""
    try:
        import uvicorn
    except ImportError:
        logger.error("Uvicorn not available. Install with: pip install uvicorn")
        return
    
    # Create translation service
    service = TranslationService(
        model_name=model_name,
        framework=framework,
        device=device
    )
    
    # Create FastAPI app
    app = create_app(service)
    if app is None:
        return
    
    logger.info(f"Starting API server on {host}:{port}")
    
    # Run the server
    uvicorn.run(app, host=host, port=port)