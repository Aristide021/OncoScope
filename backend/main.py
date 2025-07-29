"""
OncoScope API Server
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import json
import uvicorn
from datetime import datetime
import time
import uuid
import traceback
import logging
import aiohttp

from .config import settings
from .models import (
    AnalysisRequest, AnalysisResponse, HealthCheckResponse,
    ErrorResponse, MutationInput
)
from .mutation_analyzer import CancerMutationAnalyzer
from .database import init_db, get_analysis_history

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Privacy-first cancer genomics analysis powered by AI"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
start_time = time.time()
analyzer = None

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global analyzer
    logger.info("Starting OncoScope API Server...")
    
    # Initialize database
    await init_db()
    
    # Initialize mutation analyzer
    try:
        analyzer = CancerMutationAnalyzer()
        logger.info("Mutation analyzer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize mutation analyzer: {e}")
        raise
    
    logger.info("OncoScope API Server started successfully")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """System health check"""
    try:
        # Check Ollama connection
        ollama_connected = await check_ollama_connection()
        
        # Check model status
        model_loaded = await check_model_status()
        
        # Check database
        database_ready = await check_database()
        
        uptime = time.time() - start_time
        
        return HealthCheckResponse(
            status="healthy" if all([ollama_connected, model_loaded, database_ready]) else "degraded",
            ollama_connected=ollama_connected,
            model_loaded=model_loaded,
            database_ready=database_ready,
            version=settings.app_version,
            uptime_seconds=uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="error",
            ollama_connected=False,
            model_loaded=False,
            database_ready=False,
            version=settings.app_version,
            uptime_seconds=time.time() - start_time
        )

@app.post("/analyze/mutations", response_model=AnalysisResponse)
async def analyze_mutations(request: AnalysisRequest):
    """Analyze cancer mutations for clinical significance"""
    try:
        if not analyzer:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Mutation analyzer not initialized"
            )
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Log request
        logger.info(f"Starting analysis {analysis_id} for {len(request.mutations)} mutations")
        
        # Perform analysis
        start = time.time()
        analysis_result = await analyzer.analyze_mutation_list(
            mutations=request.mutations,
            include_drug_interactions=request.include_drug_interactions,
            patient_context=request.patient_context
        )
        analysis_time = time.time() - start
        
        logger.info(f"Analysis {analysis_id} completed in {analysis_time:.2f}s")
        
        # Build response
        return AnalysisResponse(
            success=True,
            analysis_id=analysis_id,
            timestamp=datetime.now(),
            **analysis_result
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    """Analyze mutations from uploaded file"""
    try:
        # Validate file type
        allowed_extensions = ['.vcf', '.csv', '.txt', '.json']
        file_ext = '.' + file.filename.split('.')[-1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Parse mutations based on file type
        mutations = parse_mutation_file(content_str, file_ext)
        
        if not mutations:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid mutations found in file"
            )
        
        # Create analysis request
        request = AnalysisRequest(mutations=mutations)
        
        # Analyze mutations
        return await analyze_mutations(request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File analysis failed: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File analysis failed: {str(e)}"
        )

@app.get("/analysis/history")
async def get_history(limit: int = 10, offset: int = 0):
    """Get analysis history"""
    try:
        history = await get_analysis_history(limit=limit, offset=offset)
        return {"success": True, "history": history}
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analysis history"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.debug_mode else None,
            error_code="INTERNAL_ERROR"
        ).dict()
    )

async def check_ollama_connection() -> bool:
    """Check if Ollama server is running"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{settings.ollama_base_url}/api/tags") as response:
                return response.status == 200
    except:
        return False

async def check_model_status() -> bool:
    """Check if OncoScope model is loaded"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{settings.ollama_base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    return any(settings.ollama_model_name in model.get('name', '') for model in models)
        return False
    except:
        return False

async def check_database() -> bool:
    """Check database connectivity"""
    try:
        # Simple database check - this would be implemented based on your DB choice
        return True
    except:
        return False

def parse_mutation_file(content: str, file_type: str) -> List[str]:
    """Parse mutations from file content"""
    mutations = []
    lines = content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        if file_type == '.json':
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    mutations = data
                elif isinstance(data, dict) and 'mutations' in data:
                    mutations = data['mutations']
                break
            except:
                pass
        else:
            # Simple parsing for CSV/TXT/VCF
            # Look for patterns like GENE:variant
            import re
            match = re.search(r'([A-Z0-9]+)[:_\s]+([cp]\.[A-Z0-9>]+)', line, re.IGNORECASE)
            if match:
                mutations.append(f"{match.group(1)}:{match.group(2)}")
    
    return mutations

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )