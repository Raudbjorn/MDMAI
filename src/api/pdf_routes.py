"""PDF processing API routes."""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from config.logging_config import get_logger
from src.pdf_processing.pipeline import PDFProcessingPipeline

logger = get_logger(__name__)

router = APIRouter(prefix="/api/pdf", tags=["pdf"])


@router.post("/upload")
async def upload_and_process_pdf(
    file: UploadFile = File(...),
    rulebook_name: str = Form(...),
    system: str = Form(...),
    source_type: str = Form("rulebook"),
    model_name: Optional[str] = Form(None),
    enable_adaptive_learning: bool = Form(True),
):
    """Upload and process a PDF file."""
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Save file with size check
    tmp_path = await _save_upload_file(file, max_size_mb=100)
    
    try:
        pipeline = PDFProcessingPipeline(
            enable_parallel=True,
            prompt_for_ollama=False,
            model_name=model_name
        )
        
        results = pipeline.process_pdf(
            pdf_path=tmp_path,
            rulebook_name=rulebook_name,
            system=system,
            source_type=source_type,
            enable_adaptive_learning=enable_adaptive_learning,
            skip_size_check=False,
            user_confirmed=True
        )
        
        # Add metadata
        results['file_info'] = {
            'filename': file.filename,
            'size_mb': round(Path(tmp_path).stat().st_size / (1024 * 1024), 2)
        }
        if model_name:
            results['embedding_model'] = model_name
        
        return results
        
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def _save_upload_file(file: UploadFile, max_size_mb: int = 100) -> str:
    """Save uploaded file to temporary location with size check."""
    max_size = max_size_mb * 1024 * 1024
    file_size = 0
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            while chunk := await file.read(8192):
                file_size += len(chunk)
                if file_size > max_size:
                    os.unlink(tmp_file.name)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Max size: {max_size_mb}MB"
                    )
                tmp_file.write(chunk)
            return tmp_file.name
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")


@router.get("/status/{task_id}")
async def get_processing_status(task_id: str):
    """Get processing status (placeholder for future async processing)."""
    return {"task_id": task_id, "status": "completed", "progress": 100}


@router.get("/recent")
async def get_recent_uploads(limit: int = 10):
    """Get recent uploads (placeholder for future implementation)."""
    return {"uploads": [], "total": 0}