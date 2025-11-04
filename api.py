"""
FastAPI application for English-Darija translation
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="English ↔ Darija Translator API",
    description="Neural machine translation API for English and Moroccan Darija",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None


class TranslationRequest(BaseModel):
    """Request model for translation"""
    text: str = Field(..., description="Text to translate", min_length=1)
    direction: str = Field(
        default="en2dar",
        description="Translation direction: 'en2dar' (English to Darija) or 'dar2en' (Darija to English)"
    )
    max_length: Optional[int] = Field(default=128, description="Maximum length of translation", ge=10, le=512)
    num_beams: Optional[int] = Field(default=5, description="Number of beams for beam search", ge=1, le=10)


class BatchTranslationRequest(BaseModel):
    """Request model for batch translation"""
    texts: List[str] = Field(..., description="List of texts to translate", min_items=1, max_items=100)
    direction: str = Field(
        default="en2dar",
        description="Translation direction: 'en2dar' or 'dar2en'"
    )
    max_length: Optional[int] = Field(default=128, ge=10, le=512)


class TranslationResponse(BaseModel):
    """Response model for translation"""
    source_text: str
    translated_text: str
    direction: str
    model_info: dict


class BatchTranslationResponse(BaseModel):
    """Response model for batch translation"""
    translations: List[dict]
    direction: str
    total_count: int


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, tokenizer, device
    
    try:
        logger.info("Loading translation model...")
        model_path = "../models/nllb-fine-tuned"  # Update this path
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        logger.info(f"Model loaded successfully on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "English ↔ Darija Translator API",
        "version": "1.0.0",
        "endpoints": {
            "translate": "/translate",
            "batch_translate": "/batch-translate",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Translate text between English and Darija
    
    Args:
        request: Translation request with text and direction
        
    Returns:
        Translation response with source and translated text
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate direction
        if request.direction not in ["en2dar", "dar2en"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid direction. Use 'en2dar' or 'dar2en'"
            )
        
        # Set language codes
        if request.direction == "en2dar":
            tokenizer.src_lang = "eng_Latn"
            tgt_lang_id = tokenizer.convert_tokens_to_ids("__ary_Arab__")
        else:
            tokenizer.src_lang = "ary_Arab"
            tgt_lang_id = tokenizer.convert_tokens_to_ids("__eng_Latn__")
        
        # Tokenize input
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Generate translation
        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=request.max_length,
                num_beams=request.num_beams,
                early_stopping=True
            )
        
        # Decode translation
        translation = tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )[0]
        
        return TranslationResponse(
            source_text=request.text,
            translated_text=translation,
            direction=request.direction,
            model_info={
                "model_name": "NLLB-200-Distilled-600M",
                "device": device,
                "max_length": request.max_length,
                "num_beams": request.num_beams
            }
        )
    
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.post("/batch-translate", response_model=BatchTranslationResponse)
async def batch_translate(request: BatchTranslationRequest):
    """
    Translate multiple texts in batch
    
    Args:
        request: Batch translation request
        
    Returns:
        Batch translation response with all translations
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate direction
        if request.direction not in ["en2dar", "dar2en"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid direction. Use 'en2dar' or 'dar2en'"
            )
        
        # Set language codes
        if request.direction == "en2dar":
            tokenizer.src_lang = "eng_Latn"
            tgt_lang_id = tokenizer.convert_tokens_to_ids("__ary_Arab__")
        else:
            tokenizer.src_lang = "ary_Arab"
            tgt_lang_id = tokenizer.convert_tokens_to_ids("__eng_Latn__")
        
        translations = []
        
        # Process in batches of 8
        batch_size = 8
        for i in range(0, len(request.texts), batch_size):
            batch = request.texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            
            # Generate translations
            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=request.max_length,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode translations
            batch_translations = tokenizer.batch_decode(
                translated_tokens,
                skip_special_tokens=True
            )
            
            # Add to results
            for source, translation in zip(batch, batch_translations):
                translations.append({
                    "source_text": source,
                    "translated_text": translation
                })
        
        return BatchTranslationResponse(
            translations=translations,
            direction=request.direction,
            total_count=len(translations)
        )
    
    except Exception as e:
        logger.error(f"Batch translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch translation failed: {str(e)}")


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "NLLB-200-Distilled-600M",
        "device": device,
        "parameters": sum(p.numel() for p in model.parameters()),
        "supported_directions": ["en2dar", "dar2en"],
        "languages": {
            "english": "eng_Latn",
            "darija": "ary_Arab"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)