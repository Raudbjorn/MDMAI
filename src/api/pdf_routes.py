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
    """
    Upload and process a PDF file with optional Ollama model selection.
    
    Args:
        file: The PDF file to upload
        rulebook_name: Name of the rulebook
        system: Game system (e.g., "D&D 5e")
        source_type: Type of source ("rulebook" or "flavor")
        model_name: Optional Ollama model name for embeddings
        enable_adaptive_learning: Whether to use adaptive learning
    
    Returns:
        Processing results and statistics
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Check file size (max 100MB)
    max_size = 100 * 1024 * 1024  # 100MB
    file_size = 0
    
    # Create temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Read and write in chunks
            while chunk := await file.read(8192):
                file_size += len(chunk)
                if file_size > max_size:
                    os.unlink(tmp_file.name)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {max_size / (1024*1024):.0f}MB"
                    )
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    
    try:
        # Initialize pipeline with specified model
        pipeline = PDFProcessingPipeline(
            enable_parallel=True,
            prompt_for_ollama=False,  # Don't prompt in API
            model_name=model_name  # Use specified model or default
        )
        
        # Process the PDF
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: asyncio.run(
                pipeline.process_pdf(
                    pdf_path=tmp_path,
                    rulebook_name=rulebook_name,
                    system=system,
                    source_type=source_type,
                    enable_adaptive_learning=enable_adaptive_learning,
                    skip_size_check=False,
                    user_confirmed=True  # Already validated size
                )
            )
        )
        
        # Add file info to results
        results['file_info'] = {
            'filename': file.filename,
            'size_bytes': file_size,
            'size_mb': round(file_size / (1024 * 1024), 2)
        }
        
        if model_name:
            results['embedding_model'] = model_name
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")


@router.get("/status/{task_id}")
async def get_processing_status(task_id: str):
    """
    Get the status of a PDF processing task.
    
    Args:
        task_id: The task ID returned from upload
    
    Returns:
        Task status and progress
    """
    # TODO: Implement task tracking for async processing
    return {
        "task_id": task_id,
        "status": "completed",
        "progress": 100,
        "message": "Processing complete"
    }


@router.get("/recent")
async def get_recent_uploads(limit: int = 10):
    """
    Get recently processed PDFs.
    
    Args:
        limit: Maximum number of results to return
    
    Returns:
        List of recent uploads with metadata
    """
    # TODO: Implement database query for recent uploads
    return {
        "uploads": [],
        "total": 0
    }