FROM apache/airflow:2.7.3-python3.11

USER airflow

# Install CPU-only PyTorch FIRST (blocks GPU version from installing later)
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy and install remaining dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt && \
    pip cache purge && \
    rm -rf /tmp/* /home/airflow/.cache

# Verify no CUDA and packages work
RUN python -c "import torch; print(f'Torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}'); import polars; import sentence_transformers; import qdrant_client; print('All deps OK')"