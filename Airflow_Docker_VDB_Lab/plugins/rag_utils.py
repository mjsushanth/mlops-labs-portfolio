"""
for my RAG pipeline - Production-minded version?? kinda.
Model: all-mpnet-base-v2 everywhere (768 dimensions)
Batch sizes: Optimized for larger model (16 for encoding, 100 for upserting)
Collection config: 768 dimensions, on-disk storage, optimized for 70k vectors
Better batching: Handles 70k chunks efficiently
Advanced search: Added method showing filtering capabilities
- Same MPNet model for encoding and querying
"""

import os
import polars as pl
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Environment detection
IS_DOCKER = os.path.exists("/opt/airflow")

if IS_DOCKER:
    DATA_PATH = Path("/opt/airflow/data/exports/sec_filings_small_full.parquet")
    QDRANT_HOST = "qdrant"
else:
    # Adjust this to your actual Windows path
    DATA_PATH = Path(__file__).parent.parent / "data" / "exports" / "sec_filings_small_full.parquet"
    QDRANT_HOST = "localhost"

logger.info(f"Running in {'Docker' if IS_DOCKER else 'Local'} environment")
logger.info(f"Data path: {DATA_PATH}")
logger.info(f"Qdrant host: {QDRANT_HOST}")



SECTION_METADATA = {
    0: {"name": "Business", "priority": "high"},
    1: {"name": "Risk Factors", "priority": "high"},
    2: {"name": "Unresolved Comments", "priority": "low", "note": "often empty"},
    3: {"name": "Properties", "priority": "medium"},
    4: {"name": "Legal Proceedings", "priority": "medium"},
    5: {"name": "Mine Safety", "priority": "low"},
    6: {"name": "Market for Stock", "priority": "medium"},
    7: {"name": "Reserved", "priority": "low", "note": "usually empty"},
    8: {"name": "MD&A", "priority": "high"},
    9: {"name": "Financial Statements", "priority": "high"},
    10: {"name": "Notes to Financials", "priority": "highest"},  # 30% of data
    11: {"name": "Market Risk", "priority": "medium"},  # Don't dismiss!
    12: {"name": "Controls", "priority": "medium"},
    13: {"name": "Unknown", "priority": "low", "note": "fragments"},
    19: {"name": "Exhibits", "priority": "low"}
}


"""
Calculate token counts (split by spaces, count words)
Add flags for long texts (>1000 chars), fragments (<20 chars), tables (multiple numbers)
Add priority from section mapping (high/medium/low)
Filter only true garbage (less than 3 tokens like "." or "NA")
"""
def load_and_enrich_data(min_tokens: int = 3, sample_size: int = None) -> pl.DataFrame:
    """
    Load ALL data from parquet with minimal filtering.
    No sampling - use entire small_full dataset.
    """
    # old solution:
    # data_path = Path("/opt/airflow/data/exports/sec_filings_small_full.parquet")
    
    df = pl.read_parquet(DATA_PATH)
    logger.info(f"Loaded {len(df)} total sentences")
    
    # DROP USELESS COLUMNS FIRST
    df = df.drop([
        "sentenceCount",     # Just an internal counter, no value
        "sentenceStartIdx",  # Character position, irrelevant for embeddings
        "sentenceEndIdx",    # Character position, irrelevant for embeddings  
        "filingDate"         # Redundant with reportDate
    ])

    # Add analytical columns for downstream processing
    df = df.with_columns([
        # Token count estimation (split by spaces) + Char length for chunking later.
        (pl.col("sentence").str.split(" ").list.len()).alias("token_count"),
        pl.col("sentence").str.len_chars().alias("char_count"),
        
        # Map section to priority level
        pl.col("section").map_elements(
            lambda x: SECTION_METADATA.get(x, {}).get("priority", "unknown"),
            return_dtype=pl.Utf8
        ).alias("section_priority"),
        
        # Flag potential data quality issues + Detect fin tables pattern.
        (pl.col("sentence").str.len_chars() > 1000).alias("is_long_text"),
        (pl.col("sentence").str.len_chars() < 20).alias("is_fragment"),
        (pl.col("sentence").str.contains(r"\d+\s+\d+\s+\d+")).alias("likely_table"),
    ])
    
    # Minimal filter - only remove true fragments
    df_filtered = df.filter(pl.col("token_count") >= min_tokens)
    
    # SIMPLE RANDOM SAMPLE
    if sample_size and sample_size < len(df_filtered):
        df_filtered = df_filtered.sample(n=sample_size, seed=42, shuffle=True)
        logger.info(f"Randomly sampled {sample_size} sentences")

    logger.info(f"Using {len(df_filtered)} sentences")
    
    return df_filtered


""" Not used rn, Retained from my proj. !!!!! 
    # Sampling strategy - stratified by section if sampling
    if n_samples and n_samples < len(df_filtered):
        # Stratified sampling to preserve section distribution
        df_filtered = (
            df_filtered
            .group_by("section")
            .apply(lambda group: group.sample(
                min(len(group), max(10, int(n_samples * len(group) / len(df_filtered)))), seed=42
            ))
        )
        logger.info(f"Sampled to {len(df_filtered)} sentences (stratified by section)")

# len(group) / len(df_filtered) = this section's percentage of total data
# n_samples * [percentage] = proportional samples from this section
"""
    

## ============================================================

"""
Group sentences by (docID, section) - never mix different documents or sections.
For each group:
    If group has <3 sentences: take whole group as one chunk
    Else: slide a 3-sentence window with stride=2 (1 sentence overlap)
    Example: Sentences [A,B,C,D,E] → Chunks [ABC], [CDE] (overlap on C)
For long chunks (>2000 chars, likely tables):
    Split at sentence boundaries into sub-chunks
    Mark with is_split=True flag
Add metadata to each chunk: company, date, section name, priority
"""
def create_smart_chunks(
    df: pl.DataFrame, 
    window_size: int = 3,
    stride: int = 2,  # Overlap for context continuity
    max_chunk_chars: int = 2000  # Handle long texts gracefully
) -> List[Dict]:
    """
    Create overlapping chunks with intelligent boundaries.
    Handles long texts without truncation.
    Args:
        df: Input dataframe
        window_size: Number of sentences per chunk
        stride: Sentences to advance (stride < window = overlap)
        max_chunk_chars: Split very long chunks (tables/lists)
    """
    chunks = []
    
    # Group by document and section (maintain context boundaries)
    for (doc_id, section), group in df.group_by(["docID", "section"]):
        # Sort by sentence order
        group = group.sort("sentenceID")
        sentences = group["sentence"].to_list()
        metadata = group.select(["name", "reportDate", "cik", "section_priority"]).row(0)
        
        # Handle empty or very short sections
        if len(sentences) < window_size:
            # Take the whole section as one chunk
            if sentences:  # Not empty
                chunk_text = " ".join(sentences)
                chunks.append({
                    "text": chunk_text,
                    "docID": doc_id,
                    "section": section,
                    "section_name": SECTION_METADATA.get(section, {}).get("name", f"Section_{section}"),
                    "company": metadata[0],
                    "reportDate": str(metadata[1]),
                    "cik": metadata[2],
                    "priority": metadata[3],
                    "chunk_id": f"{doc_id}_sec{section}_full",
                    "n_sentences": len(sentences),
                    "char_count": len(chunk_text)
                })
        else:
            # Sliding window with configurable stride
            for i in range(0, len(sentences) - window_size + 1, stride):
                chunk_sentences = sentences[i:i+window_size]
                chunk_text = " ".join(chunk_sentences)
                
                # Handle very long chunks (likely tables)
                if len(chunk_text) > max_chunk_chars:
                    # Split intelligently at sentence boundaries
                    sub_chunks = []
                    current_chunk = ""
                    
                    for sent in chunk_sentences:
                        if len(current_chunk) + len(sent) < max_chunk_chars:
                            current_chunk += " " + sent if current_chunk else sent
                        else:
                            if current_chunk:
                                sub_chunks.append(current_chunk)
                            current_chunk = sent
                    
                    if current_chunk:
                        sub_chunks.append(current_chunk)
                    
                    # Add sub-chunks
                    for j, sub_text in enumerate(sub_chunks):
                        chunks.append({
                            "text": sub_text,
                            "docID": doc_id,
                            "section": section,
                            "section_name": SECTION_METADATA.get(section, {}).get("name", f"Section_{section}"),
                            "company": metadata[0],
                            "reportDate": str(metadata[1]),
                            "cik": metadata[2],
                            "priority": metadata[3],
                            "chunk_id": f"{doc_id}_sec{section}_chunk{i}_part{j}",
                            "n_sentences": window_size,
                            "char_count": len(sub_text),
                            "is_split": True  # Flag that this was split
                        })
                else:
                    # Normal chunk
                    chunks.append({
                        "text": chunk_text,
                        "docID": doc_id,
                        "section": section,
                        "section_name": SECTION_METADATA.get(section, {}).get("name", f"Section_{section}"),
                        "company": metadata[0],
                        "reportDate": str(metadata[1]),
                        "cik": metadata[2],
                        "priority": metadata[3],
                        "chunk_id": f"{doc_id}_sec{section}_chunk{i}",
                        "n_sentences": window_size,
                        "char_count": len(chunk_text),
                        "is_split": False
                    })
    
    logger.info(f"Created {len(chunks)} chunks from {len(df)} sentences")
    
    # STATS.
    char_counts = [c["char_count"] for c in chunks]
    logger.info(f"Chunk stats - Min chars: {min(char_counts)}, Max: {max(char_counts)}, Avg: {sum(char_counts)/len(char_counts):.0f}")
    split_chunks = sum(1 for c in chunks if c.get("is_split", False))
    if split_chunks:
        logger.info(f"Split {split_chunks} long chunks (likely tables/lists)")
    
    return chunks

## ============================================================


""" Real magic.
Vectorization. 
    Load SentenceTransformer model (MiniLM - 384 dimensions)
    Check if any text exceeds model's max length (512 tokens)
    Warn about truncation but continue (model auto-truncates)
    Process in batches of 32 for memory efficiency
    Normalize vectors (unit length for cosine similarity)
    Attach embedding array to each chunk dictionary
    Return chunks with embeddings added
"""
def generate_embeddings_batch(
    chunks: List[Dict], 
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 16
) -> List[Dict]:
    """
    Generate embeddings with MPNet model.
    One-time computation, stored permanently in Qdrant.
    """
    logger.info(f"Loading embedding model: {model_name}")
    logger.info("First run downloads ~420MB model - cached for future use")
    
    model = SentenceTransformer(model_name)
    max_seq_length = model.max_seq_length
    logger.info(f"Model max sequence length: {max_seq_length} tokens")
    
    # Check for potential truncation
    texts = []
    truncated_count = 0
    
    for chunk in chunks:
        text = chunk["text"]
        estimated_tokens = len(text.split())
        
        if estimated_tokens > max_seq_length:
            truncated_count += 1
            logger.warning(f"Chunk {chunk['chunk_id']} has ~{estimated_tokens} tokens, will be truncated")
        
        texts.append(text)
    
    if truncated_count:
        logger.warning(f"{truncated_count}/{len(chunks)} chunks exceed token limit")
    
    # Generate embeddings with progress tracking
    logger.info(f"Generating embeddings for {len(texts)} chunks (batch_size={batch_size})")
    # logger.info("Expected time: ~25-35 minutes for 70k chunks")
    
    embeddings = model.encode(
        texts, 
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    # Attach embeddings to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
        chunk["embedding_dim"] = len(embedding)
    
    logger.info(f"Generated {len(chunks)} embeddings of dimension {embeddings.shape[1]}")
    return chunks



## ===========================================================


def init_qdrant_collection(
    collection_name: str = "sec_filings",
    vector_size: int = 768,  # CHANGED for MPNet!
    on_disk: bool = True  # Persist to disk
):
    """
    Initialize Qdrant collection with MPNet dimensions (768).
    Data persists in ./qdrant_storage/ folder.
    """

    # old; changed parameterized.
    # client = QdrantClient(host="qdrant", port=6333)
    client = QdrantClient(host=QDRANT_HOST, port=6333)  # Use the constant


    # Check existing collections
    try:
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if exists:
            # Get existing collection info
            existing_info = client.get_collection(collection_name)
            logger.info(f"Collection {collection_name} exists with {existing_info.points_count} points")
            
            # Check dimension match
            if existing_info.config.params.vectors.size != vector_size:
                logger.warning(f"Dimension mismatch! Existing: {existing_info.config.params.vectors.size}, New: {vector_size}")
                logger.info("Deleting existing collection...")
                client.delete_collection(collection_name)
                exists = False
            else:
                logger.info("Keeping existing collection (dimensions match)")
                return client
    except Exception as e:
        logger.error(f"Error checking collections: {e}")
        exists = False
    
    if not exists:
        # Create new collection with MPNet dimensions
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,  # 768 for MPNet
                distance=Distance.COSINE,
                on_disk=on_disk  # Store on disk for persistence
            ),
            # Optimize for search performance
            optimizers_config={
                "default_segment_number": 5,  # Better for 70k vectors
                "indexing_threshold": 20000  # Start indexing after 20k vectors
            }
        )
        logger.info(f"Created collection '{collection_name}' with {vector_size} dimensions")
    
    return client


def upsert_to_qdrant(
    chunks: List[Dict], 
    collection_name: str = "sec_filings",
    batch_size: int = 100  # Upsert in batches
):
    """
    Insert chunks with embeddings into Qdrant.
    Batched for efficiency with 70k chunks.
    
    Storage: ~200MB for 70k chunks with 768-dim vectors.
    Location: ./qdrant_storage/ (persists after Docker restart).
    """
    client = init_qdrant_collection(collection_name, vector_size=768)
    
    logger.info(f"Preparing to upsert {len(chunks)} chunks to Qdrant...")
    
    # Upsert in batches for better performance
    from itertools import islice
    
    def batch_iterator(iterable, batch_size):
        """Yield successive batches from iterable."""
        it = iter(iterable)
        while True:
            batch = list(islice(it, batch_size))
            if not batch:
                break
            yield batch
    
    total_upserted = 0
    for batch_num, chunk_batch in enumerate(batch_iterator(chunks, batch_size), 1):
        # Prepare points for this batch
        points = []
        for chunk in chunk_batch:
            # Use hash of chunk_id as integer ID (Qdrant requires int or UUID)
            point_id = abs(hash(chunk["chunk_id"])) % (10 ** 8)  # 8-digit ID
            
            point = PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload={
                    "text": chunk["text"],
                    "docID": chunk["docID"],
                    "section": chunk["section"],
                    "section_name": chunk.get("section_name", f"Section_{chunk['section']}"),
                    "company": chunk["company"],
                    "reportDate": chunk["reportDate"],
                    "cik": chunk.get("cik", ""),
                    "priority": chunk.get("priority", "unknown"),
                    "chunk_id": chunk["chunk_id"],
                    "char_count": chunk.get("char_count", len(chunk["text"])),
                    "n_sentences": chunk.get("n_sentences", 3)
                }
            )
            points.append(point)
        
        # Upsert this batch
        client.upsert(collection_name=collection_name, points=points)
        total_upserted += len(points)
        
        if batch_num % 10 == 0:  # Progress every 10 batches
            logger.info(f"  Upserted {total_upserted}/{len(chunks)} chunks...")
    
    # Final verification
    collection_info = client.get_collection(collection_name)
    logger.info(f"Collection '{collection_name}' now has {collection_info.points_count} vectors")
    logger.info(f"Data persisted in ./qdrant_storage/")
    
    return client


def test_retrieval(
    query: str = "What are the main risk factors for the business?",
    n_results: int = 5,
    section_filter: Optional[List[int]] = None, collection_name: str = "sec_filings"
):
    """
    Test semantic search with MPNet embeddings.
    Can filter by section for targeted search.
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing retrieval with query: '{query}'")
    
    # Use same model as for encoding!
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    
    # Connect to Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=6333)
    
    # Build filter if specified
    search_params = {
        "collection_name": collection_name,
        "query_vector": query_embedding,
        "limit": n_results,
        "with_payload": True,
        "with_vectors": False  # Don't return vectors (save bandwidth)
    }
    
    # Add section filter if specified
    if section_filter:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        search_params["query_filter"] = Filter(
            must=[
                FieldCondition(
                    key="section",
                    match=MatchAny(any=section_filter)
                )
            ]
        )
        logger.info(f"Filtering to sections: {section_filter}")
    
    # Perform search
    results = client.search(**search_params)
    
    # Display results
    logger.info(f"\nTop {n_results} Results:")
    logger.info("-" * 80)
    
    for i, result in enumerate(results, 1):
        score = result.score
        payload = result.payload
        
        print(f"\n{i}. [Score: {score:.4f}]")
        print(f"   Company: {payload['company']}")
        print(f"   Section: {payload['section']} ({payload.get('section_name', 'Unknown')})")
        print(f"   Date: {payload['reportDate']}")
        print(f"   Priority: {payload.get('priority', 'unknown')}")
        print(f"   Text Preview: {payload['text'][:300]}...")
        print(f"   Chunk ID: {payload['chunk_id']}")
    
    logger.info("-" * 80)
    logger.info(f"Search completed successfully")
    
    return results


def advanced_search(
    query: str,
    company_filter: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None,
    high_priority_only: bool = False,
    n_results: int = 10
):
    """
    Advanced search with multiple filters.
    Demonstrates Qdrant's filtering capabilities.
    """
    logger.info(f"\nAdvanced Search: '{query}'")
    
    # Encode query with MPNet
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    
    client = QdrantClient(host=QDRANT_HOST, port=6333)
    
    # Build complex filter
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range
    
    filter_conditions = []
    
    if company_filter:
        filter_conditions.append(
            FieldCondition(key="company", match=MatchValue(value=company_filter))
        )
        logger.info(f"  Filter: company = {company_filter}")
    
    if date_range:
        start_date, end_date = date_range
        filter_conditions.append(
            FieldCondition(
                key="reportDate",
                range=Range(gte=start_date, lte=end_date)
            )
        )
        logger.info(f"  Filter: date between {start_date} and {end_date}")
    
    if high_priority_only:
        filter_conditions.append(
            FieldCondition(
                key="priority",
                match=MatchAny(any=["high", "highest"])
            )
        )
        logger.info(f"  Filter: high priority sections only")
    
    # Apply filters if any
    search_params = {
        "collection_name": "sec_filings",
        "query_vector": query_embedding,
        "limit": n_results
    }
    
    if filter_conditions:
        search_params["query_filter"] = Filter(must=filter_conditions)
    
    results = client.search(**search_params)
    
    logger.info(f"Found {len(results)} matching results")
    return results

## ===========================================================
## STANDALONE TESTING
if __name__ == "__main__":
    """Quick test when running file directly."""
    logger.info("Testing connection to Qdrant...")
    
    try:
        client = QdrantClient(host="localhost", port=6333)  # Note: localhost when testing
        collections = client.get_collections()
        logger.info(f"Connected! Collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        logger.error(f"Cannot connect to Qdrant: {e}")
        logger.error("Make sure docker-compose services are running!")


