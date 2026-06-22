# backend.py
from fastapi import FastAPI, File, UploadFile, Form, Request, Header, HTTPException, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import requests
from dotenv import load_dotenv
from src.pipeline import process_all_documents_pipeline, query_documents_sync
from src.embed_and_index import generate_query_embedding_pinecone
from src.telemetry import save_run, load_telemetry_summary
from pinecone import Pinecone
from pydantic import BaseModel
from typing import List, Optional
import time
import asyncio
import datetime
import json
import shutil

load_dotenv()
app = FastAPI(title="HackRx Insurance API", description="API for querying insurance PDFs")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Security scheme for Bearer token authentication
security = HTTPBearer()

# Hardcoded API token - keep it simple
API_TOKEN = "552a90e441d8b2a0c195b5425dd982e0e71292568a08d2facf1ebc9434c1bcd0"

class QueryPDFRequest(BaseModel):
    documents: str  # URL to the PDF
    questions: List[str]  # List of questions to answer

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the token provided in the authorization header."""
    if credentials.scheme != "Bearer" or credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.post("/hackrx/run")
async def query_pdf(input: QueryPDFRequest, token: str = Depends(verify_token)):
    total_start_time = time.time()
    timings = {}
    pdf_url = input.documents  # Changed from pdf_url to documents
    queries = input.questions  # Changed from queries to questions
    if not pdf_url or not queries or not isinstance(queries, list):
        return JSONResponse({"error": "documents URL and questions (list) are required"}, status_code=400)

    # Create a temp directory that won't be automatically deleted
    tmpdir = tempfile.mkdtemp()
    print(f"📁 Created temporary directory: {tmpdir}")
    
    # Create a permanent directory for storing PDFs
    pdf_storage_dir = "stored_pdfs"
    os.makedirs(pdf_storage_dir, exist_ok=True)

    # Create a unique filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"input_{timestamp}.pdf"
    pdf_path = os.path.join(pdf_storage_dir, pdf_filename)

    # Download PDF
    try:
        t0 = time.time()
        r = requests.get(pdf_url)
        r.raise_for_status()
        with open(pdf_path, "wb") as f:
            f.write(r.content)
        
        # Also copy the PDF to the temporary directory for processing
        tmpdir_pdf_path = os.path.join(tmpdir, pdf_filename)
        shutil.copy2(pdf_path, tmpdir_pdf_path)
        
        timings["download"] = time.time() - t0
        print(f"📄 Downloaded PDF to: {pdf_path}")
        print(f"📄 Copied PDF to temp directory: {tmpdir_pdf_path}")
    except Exception as e:
        return JSONResponse({"error": f"Failed to download PDF: {str(e)}"}, status_code=400)
    
    # Process PDF in a sepa 
    async def process_pdf():
        pinecone_key = os.getenv("PINECONE_API_KEY")
        t0 = time.time()
        result = await process_all_documents_pipeline(
            docs_dir=tmpdir,
            pinecone_api_key=pinecone_key,
            force_reprocess=True
        )
        timings["process_and_index"] = time.time() - t0
        return result, pinecone_key
    
    # Create PDF processing task
    pdf_task = asyncio.create_task(process_pdf())
    
    # Embed all queries in a batch
    t0 = time.time()
    pinecone_key = os.getenv("PINECONE_API_KEY")

    # Process all queries in a single batch using Pinecone
    try:
        # Create Pinecone client directly instead of using get_pinecone_client
        pc = Pinecone(api_key=pinecone_key)
        model_name = "multilingual-e5-large"
        
        # Make the API call directly to ensure proper formatting
        response = pc.inference.embed(
            model=model_name,
            inputs=queries,
            parameters={"input_type": "query", "truncate": "END"}
        )
        
        # Process the response based on its structure
        if isinstance(response, dict) and 'data' in response:
            # Standard response format
            all_embeddings = [item['values'] for item in response['data']]
        elif isinstance(response, list):
            # Alternative response format
            all_embeddings = [item['values'] for item in response]
        elif hasattr(response, 'data'):
            # EmbeddingsList object format
            all_embeddings = [item['values'] for item in response.data]
        else:
            # Last resort: try to extract data directly from the response object
            try:
                # Try to convert the response to a dict
                response_dict = response.__dict__
                if 'data' in response_dict:
                    all_embeddings = [item['values'] for item in response_dict['data']]
                else:
                    # If we can't figure out the format, just use individual embedding
                    raise ValueError(f"Cannot extract embeddings from response")
            except:
                raise ValueError(f"Unexpected response format: {type(response)}")
            
        query_embedding_time = time.time() - t0
        
        # Store embedding timing information
        query_embedding_times = [query_embedding_time / len(queries)] * len(queries)
        timings["query_embedding_individual"] = query_embedding_times
        timings["query_embedding"] = query_embedding_time
        
        print(f"✅ Successfully batch-embedded {len(all_embeddings)} queries with {model_name}")
        
    except Exception as e:
        print(f"❌ Error in batch embedding: {e}")
        print(f"Response type: {type(response) if 'response' in locals() else 'Unknown'}")
        if 'response' in locals():
            print(f"Response attributes: {dir(response)}")
            if hasattr(response, 'data'):
                print(f"Response.data type: {type(response.data)}")
                if hasattr(response.data, '__len__'):
                    print(f"Response.data length: {len(response.data)}")
                    if len(response.data) > 0:
                        print(f"First item type: {type(response.data[0])}")
        
        # We'll use individual embedding as fallback since that's more reliable
        all_embeddings = []
        total_embedding_time = 0
        for query in queries:
            t_embed = time.time()
            embedding = generate_query_embedding_pinecone(query, pinecone_key)
            embed_time = time.time() - t_embed
            total_embedding_time += embed_time
            all_embeddings.append(embedding)
        
        # Update timing information for fallback case
        query_embedding_times = [total_embedding_time / len(queries)] * len(queries)
        timings["query_embedding_individual"] = query_embedding_times
        timings["query_embedding"] = total_embedding_time
    
    # Wait for PDF processing to complete
    result, pinecone_key = await pdf_task

    if not result.get("success"):
        return JSONResponse({"error": result.get("error", "Pipeline failed")}, status_code=500)

    # Process all queries together and ensure answers are in original order
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    # Process queries using the pipeline's batch processing function
    t0 = time.time()
    from src.pipeline import query_documents_batch_sync
    from src.query_processor import QueryProcessor
    
    # Initialize the QueryProcessor for cleanup later
    processor = QueryProcessor(
        pinecone_api_key=pinecone_key,
        gemini_api_key=gemini_key,
        index_name="policy-index"
    )
    timings["processor_init"] = time.time() - t0
    
    # Process all queries in parallel using asyncio with proper order preservation
    t_batch_start = time.time()
    
    print(f"🚀 Processing {len(queries)} queries in parallel mode...")
    
    # Create async tasks for all queries to run in parallel
    async def process_single_query_async(query, embedding, index):
        """Process a single query asynchronously and return result with original index"""
        t_query_start = time.time()
        
        # Run the sync function in a thread pool to avoid blocking
        import concurrent.futures
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                query_documents_sync,
                query,
                pinecone_key,
                gemini_key,
                "policy-index",
                embedding
            )
        
        query_time = time.time() - t_query_start
        return {"index": index, "result": result, "time": query_time}
    
    # Create tasks for all queries to run in parallel
    tasks = []
    for i, (query, embedding) in enumerate(zip(queries, all_embeddings)):
        task = asyncio.create_task(process_single_query_async(query, embedding, i))
        tasks.append(task)
    
    # Wait for all tasks to complete in parallel
    parallel_results = await asyncio.gather(*tasks)
    
    total_batch_time = time.time() - t_batch_start
    
    # Sort results by original index to maintain order
    parallel_results.sort(key=lambda x: x["index"])
    
    # Extract answers and individual query times in original order
    answers = [r["result"] for r in parallel_results]
    query_times = [r["time"] for r in parallel_results]
    
    print(f"✅ Completed {len(queries)} queries in parallel in {total_batch_time:.2f} seconds")
    print(f"📊 Individual query times: {[f'{t:.2f}s' for t in query_times]}")
    print(f"📊 Max query time: {max(query_times):.2f}s, Parallel execution time: {total_batch_time:.2f}s")
    
    # Clean up Pinecone index after all queries are processed
    try:
        t0 = time.time()
        if processor.index:
            # Delete all vectors from the index
            processor.index.delete(delete_all=True)
            print("✅ Successfully deleted all vectors from Pinecone index")
        timings["cleanup_index"] = time.time() - t0
    except Exception as e:
        print(f"❌ Error cleaning up Pinecone index: {e}")
        timings["cleanup_index"] = 0
        
    timings["total_execution_time"] = time.time() - total_start_time

    # Create response with individual query times and additional info
    query_timing_details = []
    for idx, (query, time_taken) in enumerate(zip(queries, query_times)):
        query_timing_details.append({
            "query_index": idx,
            "query": query[:50] + "..." if len(query) > 50 else query,  # Truncate long queries
            "time_seconds": time_taken
        })

    # Calculate combined time for all queries
    total_query_time = sum(query_times)

    # Collect all timing metrics for each step
    all_timings = {
        "download_pdf": timings.get("download", 0),
        "query_embedding": {
            "total": timings.get("query_embedding", 0),
            "individual": timings.get("query_embedding_individual", [])
        },
        "pdf_processing_and_indexing": timings.get("process_and_index", 0),
        "query_processor_initialization": timings.get("processor_init", 0),
        "query_processing": {
            "total": total_query_time,
            "average": sum(query_times) / len(query_times) if query_times else 0,
            "individual": query_times
        },
        "cleanup_index": timings.get("cleanup_index", 0),
        "total_execution_time": timings.get("total_execution_time", 0)
    }

    # Prepare the comprehensive data for logging
    comprehensive_data = {
        "answers": answers,  # This now contains the full result structure for each query
        "timings": all_timings,
        "cleanup_status": "Vectors deleted from Pinecone index",
        "temp_directory": tmpdir,  # Include the path to the preserved temporary directory
        "api_version": "2.1",  # Updated API version to reflect the new response format
        "model_info": {
            "embedding_model": "multilingual-e5-large",
            "temperature": 0.7
        }
    }
    
    # Save the request and response to a JSON file (log everything)
    try:
        # Create a logs directory if it doesn't exist
        logs_dir = "request_logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create a comprehensive log entry with all details
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "request": {
                "pdf_url": pdf_url,
                "questions": queries,
                "pdf_filename": pdf_filename,
                "pdf_stored_path": pdf_path
            },
            "comprehensive_response": comprehensive_data,  # Log everything
            "processing_summary": {
                "total_questions": len(queries),
                "total_time_seconds": timings.get("total_execution_time", 0),
                "pdf_processing_time": timings.get("process_and_index", 0),
                "query_processing_time": total_query_time,
                "parallel_execution_time": total_batch_time,
                "max_individual_query_time": max(query_times) if query_times else 0,
                "embedding_time": timings.get("query_embedding", 0)
            },
            "performance_metrics": {
                "queries_processed": len(queries),
                "parallel_speedup_ratio": (total_query_time / total_batch_time) if total_batch_time > 0 else 1,
                "individual_query_times": query_times,
                "query_timing_details": query_timing_details
            }
        }
        
        # Save to a timestamped JSON file
        log_filename = f"request_log_{timestamp}.json"
        log_path = os.path.join(logs_dir, log_filename)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Complete request and response logged to: {log_path}")
        print(f"📊 Performance: {len(queries)} queries processed in {total_batch_time:.2f}s (parallel) vs {total_query_time:.2f}s (sequential)")
        
    except Exception as e:
        print(f"⚠️ Failed to save request log: {e}")

    # Return simplified response - just the answer text for each query
    simple_answers = []
    for result in answers:
        # Extract just the answer text from the evaluation
        answer_text = result.get("evaluation", {}).get("answer", "No answer found")
        simple_answers.append(answer_text)
    
    simple_response = {
        "answers": simple_answers
    }
    
    return JSONResponse(simple_response)


# Module-level cache: pdf_hash -> list of chunks (survives across requests)
_chunk_cache: dict = {}

def _hash_file(path: str) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


async def _process_pdf_and_answer(pdf_path: str, queries: List[str]) -> JSONResponse:
    """
    Shared logic for both upload and URL endpoints.
    - Caches chunks by file hash so re-uploading the same PDF skips parse+embed.
    - Populates QueryProcessor chunk cache to eliminate Pinecone zero-vector lookups.
    """
    total_start_time = time.time()
    pinecone_key = os.getenv("PINECONE_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not pinecone_key:
        return JSONResponse({"error": "PINECONE_API_KEY not set in environment"}, status_code=500)

    file_hash = _hash_file(pdf_path)
    cached = _chunk_cache.get(file_hash)

    if cached:
        print(f"✅ Cache hit for {os.path.basename(pdf_path)} ({file_hash[:8]}) — skipping parse+embed")
        chunks = cached["chunks"]
        index_time = 0.0
    else:
        # Step 1: Parse + chunk + embed + index
        t0 = time.time()
        docs_dir = os.path.dirname(pdf_path)
        try:
            index_result = await process_all_documents_pipeline(
                docs_dir=docs_dir,
                pinecone_api_key=pinecone_key,
                force_reprocess=True
            )
        except Exception as e:
            return JSONResponse({"error": f"Indexing failed: {str(e)}"}, status_code=500)

        if not index_result.get("success"):
            return JSONResponse({"error": index_result.get("error", "Indexing pipeline failed")}, status_code=500)

        # Rebuild chunks from ordered_content so we can populate the processor cache
        from src.parse_documents import load_and_parse_documents
        from src.chunk_documents_optimized import chunk_documents_optimized

        parsed = load_and_parse_documents([pdf_path])
        transformed = []
        for doc in parsed:
            parsed_output = doc.get("parsed_output", {})
            transformed.append({
                "document_name": doc.get("document_name", ""),
                "content": parsed_output.get("content", ""),
                "ordered_content": parsed_output.get("ordered_content", []),
            })
        chunks = chunk_documents_optimized(transformed)

        # Store in module-level cache
        _chunk_cache[file_hash] = {"chunks": chunks, "filename": os.path.basename(pdf_path)}
        index_time = time.time() - t0
        print(f"✅ Indexed and cached {len(chunks)} chunks in {index_time:.1f}s")

    # Step 2: Batch-embed all queries in one API call
    t1 = time.time()
    try:
        pc = Pinecone(api_key=pinecone_key)
        response = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=queries,
            parameters={"input_type": "query", "truncate": "END"}
        )
        all_embeddings = [item.values for item in response.data] if hasattr(response, "data") else [item["values"] for item in response]
    except Exception:
        all_embeddings = [generate_query_embedding_pinecone(q, pinecone_key) for q in queries]
    embed_time = time.time() - t1

    # Step 3: Build one shared QueryProcessor and populate its chunk cache
    # This eliminates the Pinecone zero-vector fallback for adjacent chunk lookups
    from src.query_processor import QueryProcessor
    processor = QueryProcessor(
        pinecone_api_key=pinecone_key,
        gemini_api_key=gemini_key or "dummy",
        index_name="policy-index"
    )
    processor.populate_chunk_cache(chunks)

    # Step 4: Run all queries in parallel using the shared processor, track per-query time
    t2 = time.time()
    async def run_query(query: str, embedding: list, idx: int):
        loop = asyncio.get_event_loop()
        import concurrent.futures
        t_q = time.time()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool, processor.process_query_sync, query, embedding
            )
        return idx, result, round(time.time() - t_q, 2)

    tasks = [run_query(q, emb, i) for i, (q, emb) in enumerate(zip(queries, all_embeddings))]
    raw_results = await asyncio.gather(*tasks)
    raw_results.sort(key=lambda x: x[0])
    query_time = time.time() - t2
    per_query_timings = [r[2] for r in raw_results]

    # Step 5: Clean up Pinecone index (chunks stay in module cache)
    try:
        pc.Index("policy-index").delete(delete_all=True)
    except Exception as e:
        print(f"⚠️ Index cleanup failed: {e}")

    # Step 6: Build response
    answers = []
    for _, result, _ in raw_results:
        result["success"] = result.get("status") == "success"
        evaluation = result.get("evaluation", {})
        answers.append({
            "decision": evaluation.get("decision", "unclear"),
            "confidence": evaluation.get("confidence", 0.0),
            "answer": evaluation.get("answer", "No answer found"),
            "justification": evaluation.get("justification", ""),
            "relevant_clauses": evaluation.get("relevant_clauses", []),
        })

    timing = {
        "total_seconds": round(time.time() - total_start_time, 2),
        "index_seconds": round(index_time, 2) if not cached else 0,
        "embed_seconds": round(embed_time, 2),
        "query_seconds": round(query_time, 2),
        "cache_hit": bool(cached),
        "per_query_seconds": per_query_timings,
    }

    # Step 7: Save output + telemetry
    run_id = save_run(
        filename=os.path.basename(pdf_path),
        file_hash=file_hash,
        questions=queries,
        answers=answers,
        timing=timing,
        cache_hit=bool(cached),
        per_query_timings=per_query_timings,
        chunk_count=len(chunks),
    )

    return JSONResponse({
        "run_id": run_id,
        "answers": answers,
        "timing": timing,
        "indexed_file": os.path.basename(pdf_path),
    })


@app.post(
    "/hackrx/upload",
    summary="Upload a PDF and ask questions",
    description=(
        "Upload a PDF file and a newline-separated list of questions as multipart/form-data. "
        "Fields: **file** (PDF binary) and **questions** (plain text, one question per line). "
        "The file is indexed, all questions answered in parallel, and the index cleared afterwards."
    ),
    tags=["Insurance RAG"],
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file", "questions"],
                        "properties": {
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": "PDF file to ingest",
                            },
                            "questions": {
                                "type": "string",
                                "description": "One question per line",
                                "example": "What is covered under accidental death?\nAre pre-existing conditions covered?\nWhat is the waiting period for maternity?",
                            },
                        },
                    }
                }
            },
            "required": True,
        }
    },
)
async def upload_and_query(
    request: Request,
    token: str = Depends(verify_token),
):
    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse multipart form: {e}")

    file = form.get("file")
    if file is None:
        raise HTTPException(status_code=400, detail=f"Missing form field: 'file'. Fields received: {list(form.keys())}")
    if not hasattr(file, "read") or not hasattr(file, "filename"):
        raise HTTPException(status_code=400, detail="'file' must be an uploaded file, not a text field")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    questions_raw = form.get("questions")
    if questions_raw is None:
        raise HTTPException(status_code=400, detail="Missing form field: 'questions'")
    if isinstance(questions_raw, UploadFile):
        questions_raw = (await questions_raw.read()).decode("utf-8")

    query_list = [q.strip() for q in str(questions_raw).splitlines() if q.strip()]
    if not query_list:
        raise HTTPException(status_code=400, detail="At least one question is required")

    tmpdir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmpdir, file.filename)
    try:
        contents = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    return await _process_pdf_and_answer(pdf_path, query_list)


@app.get(
    "/hackrx/status",
    summary="Check index status",
    description="Returns the number of vectors currently in the Pinecone index.",
    tags=["Insurance RAG"],
)
async def index_status(token: str = Depends(verify_token)):
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY not set")
    try:
        from src.embed_and_index import get_index_stats
        stats = get_index_stats(pinecone_key, "policy-index")
        return JSONResponse(stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/hackrx/runs",
    summary="List all past runs",
    description=(
        "Returns a summary table of every recorded run: timing breakdown, "
        "confidence scores, cache hits, and decisions. Use this to compare runs over time."
    ),
    tags=["Insurance RAG"],
)
async def list_runs(token: str = Depends(verify_token)):
    records = load_telemetry_summary()
    return JSONResponse({
        "total_runs": len(records),
        "runs": records,
    })


@app.get(
    "/hackrx/runs/{run_id}",
    summary="Get full telemetry for a specific run",
    description="Returns the complete telemetry JSON for a single run by its run_id.",
    tags=["Insurance RAG"],
)
async def get_run(run_id: str, token: str = Depends(verify_token)):
    telemetry_path = os.path.join("artifacts", "telemetry", f"{run_id}.json")
    output_path = os.path.join("sample_outputs", f"{run_id}.json")

    if not os.path.exists(telemetry_path):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    with open(telemetry_path, encoding="utf-8") as f:
        telemetry = json.load(f)

    output = None
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            output = json.load(f)

    return JSONResponse({"telemetry": telemetry, "output": output})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

# To run: uvicorn backend:app --reload