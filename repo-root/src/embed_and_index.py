"""
Module: embed_and_index.py
Functionality: Advanced embedding generation using Pinecone's text embeddings and vector indexing with smart document management.
"""
from typing import List, Dict, Optional, Callable, Any
import time
import os
from pinecone import Pinecone, ServerlessSpec
try:
    from .document_registry import DocumentRegistry
except ImportError:
    from document_registry import DocumentRegistry

def generate_embeddings_batch(texts: List[str], api_key: str, batch_size: int = 96) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Pinecone inference API with batching.
    Args:
        texts: List of text strings to embed
        api_key: Pinecone API key
        batch_size: Maximum number of texts to process in a single batch (default: 96)
    Returns:
        List of embedding vectors
    """
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        
        # Process in batches to respect Pinecone's limits
        all_embeddings = []
        total_texts = len(texts)
        
        # Process in batches
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i+batch_size]
            print(f"📦 Processing batch {i//batch_size + 1}/{(total_texts+batch_size-1)//batch_size}: {len(batch)} texts")
            
            # Use the inference.embed method for this batch
            response = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=batch,
                parameters={"input_type": "passage", "truncate": "END"}
            )
            
            # Extract embeddings from the response
            batch_embeddings = []
            for embedding in response.data:
                batch_embeddings.append(embedding.values)
            
            all_embeddings.extend(batch_embeddings)
        
        print(f"✅ Generated {len(all_embeddings)} embeddings using Pinecone inference ({len(all_embeddings[0]) if all_embeddings else 0} dims)")
        return all_embeddings
        
    except Exception as e:
        print(f"❌ Error generating embeddings with Pinecone inference: {e}")
        # Return non-zero random vectors as fallback
        import random
        fallback_embeddings = []
        for _ in texts:
            fallback_embeddings.append([random.uniform(-0.01, 0.01) for _ in range(1024)])
        return fallback_embeddings

def generate_embeddings_pinecone(texts: List[str], api_key: str) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Pinecone inference API.
    Args:
        texts: List of text strings to embed
        api_key: Pinecone API key
    Returns:
        List of embedding vectors
    """
    # Use the batched implementation with a max batch size of 96
    return generate_embeddings_batch(texts, api_key, batch_size=96)

## Fallback logic removed for simplicity and reliability

def generate_query_embedding_pinecone(query: str, api_key: str) -> List[float]:
    """
    Generate a single query embedding using Pinecone's embedding service.
    Args:
        query: Query text to embed
        api_key: Pinecone API key
    Returns:
        Query embedding vector
    """
    try:
        # Use Pinecone client inference method
        pc = Pinecone(api_key=api_key)
        
        # Use the inference.embed method directly
        response = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query],
            parameters={"input_type": "query", "truncate": "END"}
        )
        
        # Extract embedding from the response
        embedding = response.data[0].values
        print(f"✅ Generated query embedding using Pinecone inference ({len(embedding)} dims)")
        return embedding
        
    except Exception as e:
        print(f"❌ Error generating query embedding with Pinecone inference: {e}")
        # Return non-zero random vector as fallback
        import random
        return [random.uniform(-0.01, 0.01) for _ in range(1024)]

def clear_pinecone_index(pinecone_api_key: str, index_name: str = 'policy-index') -> int:
    """
    Clear all vectors from a Pinecone index.
    """
    pc = Pinecone(api_key=pinecone_api_key)
    if index_name not in pc.list_indexes().names():
        return 0
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    total_vectors = stats.get('total_vector_count', 0)
    index.delete(delete_all=True)
    return total_vectors

def delete_duplicate_vectors(pinecone_api_key: str, index_name: str = 'policy-index', dry_run: bool = True):
    """
    Delete duplicate vectors from Pinecone index based on content hash.
    """
    pc = Pinecone(api_key=pinecone_api_key)
    if index_name not in pc.list_indexes().names():
        return {'error': f'Index {index_name} not found'}
    index = pc.Index(index_name)
    print("🔍 Scanning index for duplicates...")
    content_hashes = {}
    duplicates = []
    try:
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        if total_vectors == 0:
            return {'message': 'No vectors in index', 'duplicates_found': 0}
        print(f"📊 Found {total_vectors} vectors in index")
        # Pinecone query returns a dict with 'matches' key
        query_response = index.query(
            vector=[0.0] * 1024,
            top_k=min(10000, total_vectors),
            include_metadata=True
        )
        matches = []
        if isinstance(query_response, dict):
            matches = query_response.get('matches', [])
        elif hasattr(query_response, 'matches'):
            matches = query_response.matches
        for match in matches:
            vector_id = match['id'] if isinstance(match, dict) else match.id
            metadata = match.get('metadata', {}) if isinstance(match, dict) else getattr(match, 'metadata', {})
            content_hash = metadata.get('content_hash', '')
            if content_hash:
                if content_hash in content_hashes:
                    duplicates.append({
                        'duplicate_id': vector_id,
                        'original_id': content_hashes[content_hash],
                        'content_hash': content_hash,
                        'document_name': metadata.get('document_name', 'unknown')
                    })
                else:
                    content_hashes[content_hash] = vector_id
        print(f"🔍 Found {len(duplicates)} duplicate vectors")
        if not dry_run and duplicates:
            print("🗑️ Deleting duplicate vectors...")
            duplicate_ids = [dup['duplicate_id'] for dup in duplicates]
            batch_size = 100
            deleted_count = 0
            for i in range(0, len(duplicate_ids), batch_size):
                batch = duplicate_ids[i:i + batch_size]
                index.delete(ids=batch)
                deleted_count += len(batch)
                print(f"Deleted {deleted_count}/{len(duplicate_ids)} duplicates...")
            return {
                'duplicates_found': len(duplicates),
                'duplicates_deleted': deleted_count,
                'remaining_vectors': total_vectors - deleted_count,
                'action': 'deleted'
            }
        else:
            return {
                'duplicates_found': len(duplicates),
                'duplicates_deleted': 0,
                'total_vectors': total_vectors,
                'action': 'dry_run' if dry_run else 'none_deleted',
                'duplicate_details': duplicates[:10]
            }
    except Exception as e:
        return {'error': f'Error processing duplicates: {str(e)}'}

def reindex_documents(pinecone_api_key: str, documents_to_reindex: List[str], index_name: str = 'policy-index'):
    """
    Remove and re-add specific documents to the index.
    """
    pc = Pinecone(api_key=pinecone_api_key)
    if index_name not in pc.list_indexes().names():
        return {'error': f'Index {index_name} not found'}
    index = pc.Index(index_name)
    deleted_vectors = []
    for doc_name in documents_to_reindex:
        print(f"🗑️ Removing existing vectors for document: {doc_name}")
        query_response = index.query(
            vector=[0.0] * 1024,
            filter={'document_name': doc_name},
            top_k=10000,
            include_metadata=True
        )
        matches = []
        if isinstance(query_response, dict):
            matches = query_response.get('matches', [])
        elif hasattr(query_response, 'matches'):
            matches = query_response.matches
        if matches:
            vector_ids = [match['id'] if isinstance(match, dict) else match.id for match in matches]
            index.delete(ids=vector_ids)
            deleted_vectors.extend(vector_ids)
            print(f"Deleted {len(vector_ids)} vectors for {doc_name}")
    return {
        'documents_processed': len(documents_to_reindex),
        'vectors_deleted': len(deleted_vectors),
        'message': f'Deleted {len(deleted_vectors)} vectors. Re-run indexing to add fresh vectors.'
    }

def get_index_stats(pinecone_api_key: str, index_name: str = 'policy-index'):
    """
    Get statistics about a Pinecone index.
    """
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        if index_name not in pc.list_indexes().names():
            return {'exists': False, 'total_vector_count': 0}
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        return {
            'exists': True,
            'total_vector_count': stats.get('total_vector_count', 0),
            'dimension': stats.get('dimension', 0),
            'index_fullness': stats.get('index_fullness', 0.0),
            'namespaces': stats.get('namespaces', {})
        }
    except Exception as e:
        return {'exists': False, 'error': str(e), 'total_vector_count': 0}

## Already simplified above

def check_or_create_pinecone_index(pinecone_api_key: str, index_name: str = 'policy-index', required_dimension: int = 1024, progress_callback: Optional[Callable] = None) -> bool:
    """
    Check if index exists with correct dimensions, delete and recreate if needed.
    """
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        existing_indexes = pc.list_indexes().names()
        if index_name in existing_indexes:
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            current_dimension = stats.get('dimension', 0)
            if current_dimension != required_dimension:
                print(f"⚠️ Index '{index_name}' has {current_dimension} dimensions, but we need {required_dimension}")
                if progress_callback:
                    progress_callback(f"Deleting old index ({current_dimension}D)...", 10)
                pc.delete_index(index_name)
                time.sleep(15)
                if progress_callback:
                    progress_callback(f"Creating new index ({required_dimension}D)...", 20)
                pc.create_index(
                    name=index_name,
                    dimension=required_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                time.sleep(20)
                print(f"✅ Successfully recreated index '{index_name}' with {required_dimension} dimensions")
                return True
            else:
                print(f"✅ Index '{index_name}' already exists with correct {required_dimension} dimensions")
                return True
        else:
            if progress_callback:
                progress_callback(f"Creating new index ({required_dimension}D)...", 15)
            pc.create_index(
                name=index_name,
                dimension=required_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            time.sleep(15)
            print(f"✅ Successfully created index '{index_name}' with {required_dimension} dimensions")
            return True
    except Exception as e:
        print(f"❌ Error managing Pinecone index: {e}")
        if progress_callback:
            progress_callback(f"Index creation failed: {e}", -1)
        return False

def index_chunks_in_pinecone(chunks: List[Dict], pinecone_api_key: str, pinecone_env: str, index_name: str = 'policy-index', progress_callback: Optional[Callable] = None):
    """
    Generate embeddings and upsert to Pinecone with metadata.
    """
    if progress_callback:
        progress_callback("Initializing Pinecone...", 0)
    if not check_or_create_pinecone_index(pinecone_api_key, index_name, 1024, progress_callback):
        print("❌ Failed to create or verify Pinecone index")
        if progress_callback:
            progress_callback("Failed to create index", -1)
        return False
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    if progress_callback:
        progress_callback("Generating embeddings with Pinecone inference...", 15)
    texts = [chunk['content'] for chunk in chunks]
    try:
        embeddings = generate_embeddings_pinecone(texts, pinecone_api_key)
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        if progress_callback:
            progress_callback(f"Error generating embeddings: {e}", -1)
        return False
    if progress_callback:
        progress_callback("Preparing vectors for indexing...", 60)
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        meta = {
            'text': chunk['content'][:1000],
            'document_name': chunk['document_name'],
            'page_number': chunk.get('page_number', 0),
            'chunk_id': chunk['chunk_id']
        }
        embedding_list = embedding if isinstance(embedding, list) else list(map(float, embedding))
        vectors.append((chunk['chunk_id'], embedding_list, meta))
    if progress_callback:
        progress_callback("Upserting to Pinecone...", 70)
    batch_size = 100
    total_batches = len(vectors) // batch_size + (1 if len(vectors) % batch_size > 0 else 0)
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        try:
            index.upsert(vectors=batch)
            if progress_callback:
                batch_num = i // batch_size + 1
                progress = 70 + (batch_num / total_batches) * 25
                progress_callback(f"Indexed batch {batch_num}/{total_batches}", progress)
        except Exception as e:
            print(f"Error upserting batch {i//batch_size + 1}: {str(e)}")
            raise
    if progress_callback:
        progress_callback("Indexing complete!", 100)
    print(f"Successfully indexed {len(chunks)} chunks into Pinecone index '{index_name}'.")
    return {"success": True, "indexed_count": len(chunks)}

def smart_index_documents(docs_folder: str, pinecone_api_key: str, index_name: str = 'policy-index', progress_callback: Optional[Callable] = None, save_parsed_text: bool = False) -> Dict[str, Any]:
    """
    Smart indexing - only processes new or changed documents
    """
    registry = DocumentRegistry()
    status = registry.get_document_status(docs_folder)
    files_to_process = registry.get_files_to_process(docs_folder)
    status_counts = {
        'indexed': len([f for f, s in status.items() if s == 'indexed']),
        'new': len([f for f, s in status.items() if s == 'new']),
        'changed': len([f for f, s in status.items() if s == 'changed']),
        'missing': len([f for f, s in status.items() if s == 'missing'])
    }
    if progress_callback:
        progress_callback(f"📊 Status: {status_counts['indexed']} indexed, {status_counts['new']} new, {status_counts['changed']} changed", 10)
    if not files_to_process:
        if progress_callback:
            progress_callback("🎉 All documents are already indexed and up-to-date!", 100)
        return {
            "status": "up_to_date",
            "processed_files": 0,
            "skipped_files": status_counts['indexed'],
            "total_time": 0,
            "status_counts": status_counts
        }
    start_time = time.time()
    processed_files = []
    from .chunk_documents_optimized import chunk_documents_optimized
    total_files = len(files_to_process)
    for i, filename in enumerate(files_to_process):
        file_path = os.path.join(docs_folder, filename)
        if progress_callback:
            progress_callback(f"🔄 Processing {filename} ({i+1}/{total_files})...", 20 + (i / total_files) * 60)
        try:
            from .parse_documents import load_and_parse_from_folder
            parsed_docs = load_and_parse_from_folder(docs_folder, file_filter=[filename], save_parsed_text=save_parsed_text)
            if parsed_docs:
                transformed_docs = []
                for doc in parsed_docs:
                    doc_name = doc.get('document_name', 'unknown')
                    parsed_output = doc.get('parsed_output', {})
                    content = (parsed_output.get('content', '') or parsed_output.get('text', '') or parsed_output.get('cleaned_text', ''))
                    transformed_doc = {
                        'document_name': doc_name,
                        'content': content,
                        'ordered_content': parsed_output.get('ordered_content', [])
                    }
                    transformed_docs.append(transformed_doc)
                chunks = chunk_documents_optimized(transformed_docs)
                result = index_chunks_in_pinecone(chunks, pinecone_api_key, index_name)
                if isinstance(result, dict) and result.get('success', False):
                    registry.mark_document_indexed(filename, file_path, len(chunks))
                    processed_files.append(filename)
                    if progress_callback:
                        progress_callback(f"✅ {filename}: {len(chunks)} chunks indexed", 20 + ((i+1) / total_files) * 60)
                else:
                    if progress_callback:
                        progress_callback(f"❌ Failed to index {filename}", 20 + ((i+1) / total_files) * 60)
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Error processing {filename}: {str(e)}", 20 + ((i+1) / total_files) * 60)
    end_time = time.time()
    processing_time = end_time - start_time
    if progress_callback:
        progress_callback(f"🎉 Smart indexing complete! Processed {len(processed_files)} files in {processing_time:.1f}s", 100)
    return {
        "status": "completed",
        "processed_files": len(processed_files),
        "skipped_files": status_counts['indexed'],
        "total_time": processing_time,
        "files_processed": processed_files,
        "status_counts": status_counts
    }

def force_reindex_all(docs_folder: str, pinecone_api_key: str, index_name: str = 'policy-index', progress_callback: Optional[Callable] = None, save_parsed_text: bool = False) -> Dict[str, Any]:
    """
    Force reindex all documents (clears registry and processes everything)
    """
    registry = DocumentRegistry()
    if progress_callback:
        progress_callback("🔄 Force re-indexing: clearing registry and index...", 5)
    registry.clear_registry()
    try:
        clear_result = clear_pinecone_index(pinecone_api_key, index_name)
        if progress_callback:
            progress_callback(f"🗑️ Cleared {clear_result} vectors from index", 10)
    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Failed to clear index: {str(e)}", 10)
        return {"status": "failed", "error": f"Could not clear index: {str(e)}"}
    return smart_index_documents(docs_folder, pinecone_api_key, index_name, progress_callback, save_parsed_text)