"""
First ver code was so dirty, a bit long. 
Lets refactor with generic helpers. Rewriting.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import json
import polars as pl
import sys
sys.path.insert(0, '/opt/airflow/plugins')

from rag_utils import (
    load_and_enrich_data, 
    create_smart_chunks, 
    generate_embeddings_batch, 
    upsert_to_qdrant,
    test_retrieval
)


# ============= GENERIC HELPERS =============
def save_data(data, path, format='json'):
    """Generic save function for any data type"""
    if format == 'json':
        with open(path, 'w') as f:
            json.dump(data, f)
    elif format == 'parquet':
        data.write_parquet(path)
    print(f"Saved {format} to {path}")
    return path

def load_data(path, format='json'):
    """Generic load function"""
    if format == 'json':
        with open(path, 'r') as f:
            return json.load(f)
    elif format == 'parquet':
        return pl.read_parquet(path)

def push_metric(context, key, value):
    """Push metrics to Airflow XCom"""
    context['task_instance'].xcom_push(key=key, value=value)
    print(f"Metric: {key} = {value}")

# ============= DAG CONFIGURATION =============
default_args = {
    'owner': 'joel',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'sec_rag_pipeline_clean',
    default_args=default_args,
    description='Clean RAG pipeline with helpers - 500 sentences',
    schedule_interval=None,
    catchup=False,
    tags=['rag', 'demo', 'clean'],
)

# =============================================================================================

# extract_data -> create_chunks -> embed_chunks -> store_vectors -> validate_pipeline

def extract_data(**context):
    """Task 1: Load and filter data"""
    df = load_and_enrich_data(min_tokens=3, sample_size=500)
    save_data(df, "/tmp/filtered_data.parquet", format='parquet')
    push_metric(context, 'row_count', len(df))
    push_metric(context, 'column_count', len(df.columns))
    return "Extract complete"

def create_chunks(**context):
    """Task 2: Create overlapping chunks"""
    df = load_data("/tmp/filtered_data.parquet", format='parquet')
    chunks = create_smart_chunks(df, window_size=3, stride=2)
    save_data(chunks, "/tmp/chunks.json", format='json')
    push_metric(context, 'chunk_count', len(chunks))
    return "Chunking complete"

def embed_chunks(**context):
    """Task 3: Generate embeddings with MPNet"""
    chunks = load_data("/tmp/chunks.json", format='json')
    chunks_embedded = generate_embeddings_batch(
        chunks, 
        model_name="sentence-transformers/all-mpnet-base-v2",  # Keep MPNet as you wanted
        batch_size=16
    )
    save_data(chunks_embedded, "/tmp/chunks_embedded.json", format='json')
    push_metric(context, 'embeddings_created', len(chunks_embedded))
    return "Embeddings complete"

def store_vectors(**context):
    """Task 4: Store in Qdrant vector database"""
    chunks = load_data("/tmp/chunks_embedded.json", format='json')
    upsert_to_qdrant(chunks, collection_name="sec_demo")
    push_metric(context, 'vectors_stored', len(chunks))
    return "Storage complete"

def validate_pipeline(**context):
    """Task 5: Test retrieval to validate pipeline"""
    results = test_retrieval(
        query="What are the main business risks?",
        n_results=3,
        collection_name="sec_demo"
    )
    
    if len(results) > 0:
        push_metric(context, 'validation_results', len(results))
        return "Pipeline validated successfully"
    else:
        raise ValueError("No search results - pipeline failed")

# =============================================================================================
# ============= TASK DEFINITIONS =============
t1 = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

t2 = PythonOperator(
    task_id='create_chunks',
    python_callable=create_chunks,
    dag=dag,
)

t3 = PythonOperator(
    task_id='embed_chunks',
    python_callable=embed_chunks,
    dag=dag,
)

t4 = PythonOperator(
    task_id='store_vectors',
    python_callable=store_vectors,
    dag=dag,
)

t5 = PythonOperator(
    task_id='validate_pipeline',
    python_callable=validate_pipeline,
    dag=dag,
)

# ============= PIPELINE FLOW =============
t1 >> t2 >> t3 >> t4 >> t5