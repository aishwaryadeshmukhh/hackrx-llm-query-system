# backend_2.py - True Multithreading Implementation
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
from pinecone import Pinecone
from pydantic import BaseModel
from typing import List, Optional
import time
import asyncio
import datetime
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

load_dotenv()
app = FastAPI(title="HackRx Insurance API - Multithreading", description="API for querying insurance PDFs with true multithreading")

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

def embed_single_query_thread(query: str, pinecone_key: str, index: int, results_queue: queue.Queue):
    """Thread function to embed a single query"""
    try:
        thread_id = threading.get_ident()
        print(f"🧵 Thread {thread_id}: Starting embedding for query {index+1}")
        
        start_time = time.time()
        embedding = generate_query_embedding_pinecone(query, pinecone_key)
        end_time = time.time()
        
        result = {
            'index': index,
            'query': query,
            'embedding': embedding,
            'time': end_time - start_time,
            'thread_id': thread_id,
            'success': True
        }
        results_queue.put(result)
        print(f"✅ Thread {thread_id}: Completed embedding for query {index+1} in {end_time - start_time:.2f}s")
        
    except Exception as e:
        error_result = {
            'index': index,
            'query': query,
            'embedding': None,
            'time': 0,
            'thread_id': threading.get_ident(),
            'success': False,
            'error': str(e)
        }
        results_queue.put(error_result)
        print(f"❌ Thread {threading.get_ident()}: Failed embedding for query {index+1}: {e}")

def process_single_query_thread(query: str, embedding: list, pinecone_key: str, gemini_key: str, index: int, results_queue: queue.Queue):
    """Thread function to process a single query with similarity search and evaluation"""
    try:
        thread_id = threading.get_ident()
        print(f"🧵 Thread {thread_id}: Starting query processing for query {index+1}")
        
        start_time = time.time()
        
        # Process query using the existing sync function
        result = query_documents_sync(
            query=query,
            pinecone_api_key=pinecone_key,
            gemini_api_key=gemini_key,
            index_name="policy-index",
            query_embedding=embedding
        )
        
        end_time = time.time()
        
        query_result = {
            'index': index,
            'query': query,
            'result': result,
            'time': end_time - start_time,
            'thread_id': thread_id,
            'success': True
        }
        results_queue.put(query_result)
        print(f"✅ Thread {thread_id}: Completed query processing for query {index+1} in {end_time - start_time:.2f}s")
        
    except Exception as e:
        error_result = {
            'index': index,
            'query': query,
            'result': None,
            'time': 0,
            'thread_id': threading.get_ident(),
            'success': False,
            'error': str(e)
        }
        results_queue.put(error_result)
        print(f"❌ Thread {threading.get_ident()}: Failed query processing for query {index+1}: {e}")

def batch_embed_queries_multithread(queries: List[str], pinecone_key: str, max_workers: int = 10):
    """Embed multiple queries using multithreading"""
    print(f"🚀 Starting multithreaded embedding for {len(queries)} queries with {max_workers} workers")
    
    start_time = time.time()
    results_queue = queue.Queue()
    
    # Create and start threads
    threads = []
    for i, query in enumerate(queries):
        thread = threading.Thread(
            target=embed_single_query_thread,
            args=(query, pinecone_key, i, results_queue)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Collect results
    embedding_results = []
    while not results_queue.empty():
        embedding_results.append(results_queue.get())
    
    # Sort by index to maintain order
    embedding_results.sort(key=lambda x: x['index'])
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Extract embeddings and timing info
    all_embeddings = []
    individual_times = []
    thread_info = []
    
    for result in embedding_results:
        if result['success']:
            all_embeddings.append(result['embedding'])
            individual_times.append(result['time'])
            thread_info.append(f"Query {result['index']+1}: Thread {result['thread_id']}")
        else:
            print(f"❌ Failed to embed query {result['index']+1}: {result.get('error', 'Unknown error')}")
            # Use fallback embedding for failed queries
            try:
                fallback_embedding = generate_query_embedding_pinecone(result['query'], pinecone_key)
                all_embeddings.append(fallback_embedding)
                individual_times.append(0)  # Mark as fallback
                thread_info.append(f"Query {result['index']+1}: Fallback (main thread)")
            except:
                # If fallback also fails, use dummy embedding
                all_embeddings.append([0.0] * 1024)  # Dummy embedding
                individual_times.append(0)
                thread_info.append(f"Query {result['index']+1}: Failed")
    
    print(f"✅ Completed multithreaded embedding in {total_time:.2f}s")
    print(f"📊 Thread assignments: {thread_info}")
    print(f"📊 Individual embedding times: {[f'{t:.2f}s' for t in individual_times]}")
    
    return all_embeddings, individual_times, total_time, thread_info

def batch_process_queries_multithread(queries: List[str], embeddings: List[list], pinecone_key: str, gemini_key: str, max_workers: int = 15):
    """Process multiple queries using multithreading"""
    print(f"🚀 Starting multithreaded query processing for {len(queries)} queries with {max_workers} workers")
    
    start_time = time.time()
    results_queue = queue.Queue()
    
    # Limit concurrent threads to avoid overwhelming APIs
    semaphore = threading.Semaphore(max_workers)
    
    def thread_wrapper(query, embedding, index):
        with semaphore:
            process_single_query_thread(query, embedding, pinecone_key, gemini_key, index, results_queue)
    
    # Create and start threads
    threads = []
    for i, (query, embedding) in enumerate(zip(queries, embeddings)):
        thread = threading.Thread(
            target=thread_wrapper,
            args=(query, embedding, i)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Collect results
    query_results = []
    while not results_queue.empty():
        query_results.append(results_queue.get())
    
    # Sort by index to maintain order
    query_results.sort(key=lambda x: x['index'])
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Extract results and timing info
    answers = []
    individual_times = []
    thread_info = []
    
    for result in query_results:
        if result['success']:
            answers.append(result['result'])
            individual_times.append(result['time'])
            thread_info.append(f"Query {result['index']+1}: Thread {result['thread_id']}")
        else:
            # Create error response for failed queries
            error_response = {
                "query": result['query'],
                "search_results": [],
                "evaluation": {
                    "answer": f"Query processing failed: {result.get('error', 'Unknown error')}",
                    "search_method": "error",
                    "error": True
                },
                "api_status": {"error": True},
                "status": "error",
                "success": False
            }
            answers.append(error_response)
            individual_times.append(0)
            thread_info.append(f"Query {result['index']+1}: Failed")
    
    print(f"✅ Completed multithreaded query processing in {total_time:.2f}s")
    print(f"📊 Thread assignments: {thread_info}")
    print(f"📊 Individual query times: {[f'{t:.2f}s' for t in individual_times]}")
    print(f"📊 Max query time: {max(individual_times):.2f}s, Parallel execution time: {total_time:.2f}s")
    
    return answers, individual_times, total_time, thread_info

@app.post("/hackrx/run")
async def query_pdf(input: QueryPDFRequest, token: str = Depends(verify_token)):
    total_start_time = time.time()
    timings = {}
    pdf_url = input.documents
    queries = input.questions
    
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
    
    # Process PDF (this still uses asyncio as it's part of the existing pipeline)
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
    
    # Get API keys
    pinecone_key = os.getenv("PINECONE_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    # Embed all queries using multithreading
    print(f"🧵 Starting multithreaded embedding for {len(queries)} queries...")
    embedding_start = time.time()
    
    # Determine optimal number of threads (max 10 for embedding to avoid rate limits)
    max_embedding_workers = min(10, len(queries))
    all_embeddings, embedding_times, total_embedding_time, embedding_threads = batch_embed_queries_multithread(
        queries, pinecone_key, max_embedding_workers
    )
    
    timings["query_embedding"] = total_embedding_time
    timings["query_embedding_individual"] = embedding_times
    timings["embedding_thread_info"] = embedding_threads
    
    print(f"✅ Completed multithreaded embedding in {total_embedding_time:.2f}s")
    
    # Wait for PDF processing to complete
    result, pinecone_key = await pdf_task

    if not result.get("success"):
        return JSONResponse({"error": result.get("error", "Pipeline failed")}, status_code=500)

    # Initialize QueryProcessor for cleanup (single instance)
    from src.query_processor import QueryProcessor
    t0 = time.time()
    processor = QueryProcessor(
        pinecone_api_key=pinecone_key,
        gemini_api_key=gemini_key,
        index_name="policy-index"
    )
    timings["processor_init"] = time.time() - t0
    
    # Process all queries using multithreading
    print(f"🧵 Starting multithreaded query processing for {len(queries)} queries...")
    
    # Determine optimal number of threads (max 5 for query processing to avoid overwhelming LLM APIs)
    max_query_workers = min(5, len(queries))
    answers, query_times, total_batch_time, query_threads = batch_process_queries_multithread(
        queries, all_embeddings, pinecone_key, gemini_key, max_query_workers
    )
    
    timings["query_processing_threads"] = query_threads
    
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
            "query": query[:50] + "..." if len(query) > 50 else query,
            "time_seconds": time_taken
        })

    # Calculate combined time for all queries
    total_query_time = sum(query_times)

    # Collect all timing metrics for each step
    all_timings = {
        "download_pdf": timings.get("download", 0),
        "query_embedding": {
            "total": timings.get("query_embedding", 0),
            "individual": timings.get("query_embedding_individual", []),
            "thread_assignments": timings.get("embedding_thread_info", [])
        },
        "pdf_processing_and_indexing": timings.get("process_and_index", 0),
        "query_processor_initialization": timings.get("processor_init", 0),
        "query_processing": {
            "total": total_query_time,
            "parallel_execution_time": total_batch_time,
            "average": sum(query_times) / len(query_times) if query_times else 0,
            "individual": query_times,
            "thread_assignments": timings.get("query_processing_threads", [])
        },
        "cleanup_index": timings.get("cleanup_index", 0),
        "total_execution_time": timings.get("total_execution_time", 0)
    }

    # Prepare the comprehensive data for logging
    comprehensive_data = {
        "answers": answers,
        "timings": all_timings,
        "cleanup_status": "Vectors deleted from Pinecone index",
        "temp_directory": tmpdir,
        "api_version": "2.2",  # Updated version for multithreading
        "model_info": {
            "embedding_model": "multilingual-e5-large",
            "temperature": 0.7,
            "multithreading": True,
            "max_embedding_workers": max_embedding_workers,
            "max_query_workers": max_query_workers
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
            "comprehensive_response": comprehensive_data,
            "processing_summary": {
                "total_questions": len(queries),
                "total_time_seconds": timings.get("total_execution_time", 0),
                "pdf_processing_time": timings.get("process_and_index", 0),
                "embedding_time": total_embedding_time,
                "query_processing_time": total_query_time,
                "parallel_execution_time": total_batch_time,
                "max_individual_query_time": max(query_times) if query_times else 0
            },
            "multithreading_metrics": {
                "queries_processed": len(queries),
                "embedding_workers": max_embedding_workers,
                "query_workers": max_query_workers,
                "embedding_speedup_ratio": (sum(embedding_times) / total_embedding_time) if total_embedding_time > 0 else 1,
                "query_speedup_ratio": (total_query_time / total_batch_time) if total_batch_time > 0 else 1,
                "individual_embedding_times": embedding_times,
                "individual_query_times": query_times,
                "embedding_thread_assignments": embedding_threads,
                "query_thread_assignments": query_threads,
                "query_timing_details": query_timing_details
            }
        }
        
        # Save to a timestamped JSON file
        log_filename = f"multithread_log_{timestamp}.json"
        log_path = os.path.join(logs_dir, log_filename)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Complete request and response logged to: {log_path}")
        print(f"🧵 Multithreading Performance:")
        print(f"   📊 Embedding: {len(queries)} queries in {total_embedding_time:.2f}s with {max_embedding_workers} threads")
        print(f"   📊 Query Processing: {len(queries)} queries in {total_batch_time:.2f}s with {max_query_workers} threads")
        print(f"   📊 Embedding Speedup: {(sum(embedding_times) / total_embedding_time):.2f}x")
        print(f"   📊 Query Speedup: {(total_query_time / total_batch_time):.2f}x")
        
    except Exception as e:
        print(f"⚠️ Failed to save request log: {e}")

    # Return response with answers, vectors, similarity results and timing information
    simple_answers = []
    similarity_vectors = []
    
    for result in answers:
        # Extract just the answer text from the evaluation
        answer_text = result.get("evaluation", {}).get("answer", "No answer found")
        simple_answers.append(answer_text)
        
        # Extract similarity search results with scores
        search_results = result.get("search_results", [])
        query_similarity_data = []
        
        for search_result in search_results:
            similarity_info = {
                "id": search_result.get("id", ""),
                "similarity_score": search_result.get("score", 0.0),
                "hybrid_score": search_result.get("hybrid_score", search_result.get("score", 0.0)),
                "text": search_result.get("text", ""),
                "document_name": search_result.get("document_name", ""),
                "page_number": search_result.get("page_number", 1)
            }
            query_similarity_data.append(similarity_info)
        
        similarity_vectors.append(query_similarity_data)
    
    # Prepare the response with vectors and timing
    response_data = {
        "answers": simple_answers,# Query embeddings used for search
        "similarity_vectors": similarity_vectors,  # Similarity search results with scores
        "total_time_taken": timings.get("total_execution_time", 0),
        "timing_breakdown": {
            "pdf_download": timings.get("download", 0),
            "pdf_processing": timings.get("process_and_index", 0),
            "embedding_time": total_embedding_time,
            "query_processing_time": total_batch_time,
            "cleanup_time": timings.get("cleanup_index", 0)
        },
        "multithreading_performance": {
            "embedding_workers": max_embedding_workers,
            "query_workers": max_query_workers,
            "embedding_speedup": (sum(embedding_times) / total_embedding_time) if total_embedding_time > 0 else 1,
            "query_speedup": (total_query_time / total_batch_time) if total_batch_time > 0 else 1,
            "individual_embedding_times": embedding_times,
            "individual_query_times": query_times
        },
        "vector_info": {
            "total_queries": len(all_embeddings),
            "query_embedding_dimension": len(all_embeddings[0]) if all_embeddings else 0,
            "embedding_model": "multilingual-e5-large",
            "total_similarity_results": sum(len(sv) for sv in similarity_vectors)
        }
    }
    
    return JSONResponse(response_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_2:app", host="0.0.0.0", port=8001, reload=True)

# To run: uvicorn backend_2:app --host 0.0.0.0 --port 8001 --reload
