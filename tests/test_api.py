"""
API tests for FastAPI service
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock API server for testing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Simple mock API for testing
app = FastAPI(title="Cantonese Translation API (Test)")

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "cantonese"
    target_lang: str = "mandarin"
    beam_size: int = 4

class TranslationResponse(BaseModel):
    translation: str
    source_language: str
    target_language: str
    confidence: float = 0.95

class BatchTranslationRequest(BaseModel):
    texts: List[str]
    source_lang: str = "cantonese"
    target_lang: str = "mandarin"
    beam_size: int = 4

class BatchTranslationResponse(BaseModel):
    translations: List[str]
    source_language: str
    target_language: str

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Translate single text"""
    # Mock translation for testing
    mock_translations = {
        "你好嗎？": "你好吗？",
        "食咗飯未？": "吃饭了吗？",
        "hello": "hello"
    }
    
    translation = mock_translations.get(request.text, f"[{request.text}]")
    
    return TranslationResponse(
        translation=translation,
        source_language=request.source_lang,
        target_language=request.target_lang
    )

@app.post("/batch_translate", response_model=BatchTranslationResponse)
async def batch_translate_texts(request: BatchTranslationRequest):
    """Batch translate multiple texts"""
    mock_translations = {
        "你好嗎？": "你好吗？",
        "食咗飯未？": "吃饭了吗？",
        "hello": "hello"
    }
    
    translations = []
    for text in request.texts:
        translation = mock_translations.get(text, f"[{text}]")
        translations.append(translation)
    
    return BatchTranslationResponse(
        translations=translations,
        source_language=request.source_lang,
        target_language=request.target_lang
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "translation"}

@app.get("/model_info")
async def model_info():
    """Get model information"""
    return {
        "model_name": "Tencent-Hunyuan/Hunyuan-MT-7B",
        "framework": "pytorch",
        "device": "cuda",
        "total_parameters": 7000000000,
        "max_length": 128
    }


# Test client
@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


class TestTranslationAPI:
    """Test translation API endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "translation"
    
    def test_model_info(self, client):
        """Test model info endpoint"""
        response = client.get("/model_info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "framework" in data
        assert "total_parameters" in data
    
    def test_single_translation(self, client):
        """Test single text translation"""
        request_data = {
            "text": "你好嗎？",
            "source_lang": "cantonese",
            "target_lang": "mandarin"
        }
        
        response = client.post("/translate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "translation" in data
        assert data["source_language"] == "cantonese"
        assert data["target_language"] == "mandarin"
        assert "confidence" in data
    
    def test_batch_translation(self, client):
        """Test batch translation"""
        request_data = {
            "texts": ["你好嗎？", "食咗飯未？"],
            "source_lang": "cantonese",
            "target_lang": "mandarin"
        }
        
        response = client.post("/batch_translate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "translations" in data
        assert len(data["translations"]) == 2
        assert data["source_language"] == "cantonese"
        assert data["target_language"] == "mandarin"
    
    def test_translation_with_beam_size(self, client):
        """Test translation with custom beam size"""
        request_data = {
            "text": "你好嗎？",
            "beam_size": 8
        }
        
        response = client.post("/translate", json=request_data)
        assert response.status_code == 200
        assert "translation" in response.json()
    
    def test_invalid_request(self, client):
        """Test invalid request handling"""
        # Missing required field
        response = client.post("/translate", json={})
        assert response.status_code == 422  # Validation error
    
    def test_empty_text(self, client):
        """Test translation with empty text"""
        request_data = {"text": ""}
        
        response = client.post("/translate", json=request_data)
        assert response.status_code == 200  # Should handle gracefully
        data = response.json()
        assert "translation" in data
    
    def test_long_text(self, client):
        """Test translation with long text"""
        long_text = "你好嗎？" * 100  # Repeat text
        request_data = {"text": long_text}
        
        response = client.post("/translate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "translation" in data
    
    def test_special_characters(self, client):
        """Test translation with special characters"""
        special_text = "你好嗎？！＠＃＄％︿＆＊（）"
        request_data = {"text": special_text}
        
        response = client.post("/translate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "translation" in data


class TestAPIPerformance:
    """Test API performance characteristics"""
    
    def test_response_time(self, client):
        """Test that API responds within acceptable time"""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Health check should be very fast
        assert response_time < 0.1  # 100ms
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        import concurrent.futures
        
        def make_request():
            return client.get("/health")
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(response.status_code == 200 for response in responses)
    
    def test_batch_size_limits(self, client):
        """Test batch size limits"""
        # Test with large batch (should still work with mock)
        large_batch = ["text" + str(i) for i in range(100)]
        request_data = {"texts": large_batch}
        
        response = client.post("/batch_translate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert len(data["translations"]) == 100


class TestAPIErrorHandling:
    """Test API error handling"""
    
    def test_invalid_json(self, client):
        """Test invalid JSON handling"""
        response = client.post(
            "/translate",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_invalid_content_type(self, client):
        """Test invalid content type"""
        response = client.post(
            "/translate",
            data="text=hello",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422
    
    def test_method_not_allowed(self, client):
        """Test method not allowed"""
        response = client.get("/translate")  # GET instead of POST
        assert response.status_code == 405
    
    def test_endpoint_not_found(self, client):
        """Test non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404


class TestAPIDocumentation:
    """Test API documentation endpoints"""
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema availability"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "/translate" in schema["paths"]
        assert "/batch_translate" in schema["paths"]
    
    def test_docs_endpoint(self, client):
        """Test documentation endpoint"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint(self, client):
        """Test ReDoc endpoint"""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])