
### Debugging Docker and Airflow:
- **For Airflow logs**: `docker-compose logs airflow-scheduler --tail=200`
- **To see all logs**: `docker-compose logs --tail=1000` (or `--follow` for live tailing)
- From terminal, see task logs: `docker exec -it airflow_docker_vdb_lab-airflow-scheduler-1 cat /opt/airflow/logs/dag_id=sec_rag_pipeline_clean/run_id=manual__2025-10-06T12:19:14.053827+00:00/task_id=extract_data/attempt=2.log`
- Other: `docker exec -it airflow_docker_vdb_lab-airflow-scheduler-1 ls -lh /tmp/filtered_data.parquet` 

- Docker Restart: `docker-compose restart airflow-scheduler airflow-webserver`

- Secret Gen: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
- Persistence: `docker exec -it airflow_docker_vdb_lab-qdrant-1 ls -lh /qdrant/storage/collections/sec_demo/`

### Cleaning up Docker resources:
- `docker-compose down`
- `docker rmi airflow_docker_vdb_lab-airflow-scheduler:latest`
- `docker rmi airflow_docker_vdb_lab-airflow-webserver:latest`
- `docker rmi airflow_docker_vdb_lab-airflow-init:latest`
- Docker Dangling images: `docker image prune -f`
- Check: `docker images`

- `docker builder prune -f` better, safer.
- docker system prune -a --volumes (Caution: removes all unused data, including volumes)

### More complex: Reclaiming space.
- Docker Desktop uses a virtual disk file (ext4.vhdx)
- That file doesn't shrink automatically even when Docker deletes data
- `docker system df` to see disk usage:
```
                                                                                                                                                   TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          6         2         3.39GB    3.024GB (89%)
Containers      3         0         10.53MB   10.53MB (100%)
Local Volumes   1         0         69.38MB   69.38MB (100%)
Build Cache     10        0         0B        0B
```

- Other: Select "Clean / Purge data" in Docker Desktop settings > Resources > Advanced.
- `docker-compose build --no-cache`

- docker images
- docker images | Select-String "airflow_docker_vdb_lab"

- Inspect what's inside the bloated image:
    - `docker run --rm -it airflow_docker_vdb_lab-airflow-scheduler:latest sh -c "du -sh /* 2>/dev/null | sort -h"`

- `docker run --rm --entrypoint /bin/bash airflow_docker_vdb_lab-airflow-scheduler:latest -c "du -sh /* 2>/dev/null | sort -h"`
- `docker run --rm --entrypoint /bin/bash airflow_docker_vdb_lab-airflow-scheduler:latest -c "du -sh /home/airflow/.local/lib/python3.11/site-packages/* 2>/dev/null | sort -h | tail -20"`

- docker run --rm --entrypoint /bin/bash airflow-rag-custom:latest -c "du -sh /* 2>/dev/null | sort -h"


### Follow cleanups + Quick perform build and compose up:
docker-compose build --no-cache
docker images | Select-String "airflow_docker_vdb_lab"
docker-compose up -d
docker exec -it airflow_docker_vdb_lab-airflow-scheduler-1 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"




## ==================================================
## Complete nuclear cleanup and rebuild:

1. Stop everything
docker-compose down -v

2. Remove ALL Airflow images (including the conflicting one)
docker rmi -f airflow-rag-custom:latest
docker rmi -f airflow_docker_vdb_lab-airflow-scheduler:latest
docker rmi -f airflow_docker_vdb_lab-airflow-webserver:latest
docker rmi -f airflow_docker_vdb_lab-airflow-init:latest

3. Remove dangling images and build cache
docker image prune -af
<!-- docker builder prune -af -->

# 3.5. Shutdown WSL to release disk (optional but helpful)
wsl --shutdown
Start-Sleep -Seconds 15


4. Verify cleanup
docker images | Select-String "airflow"
# Should show NOTHING with "airflow" in the name

5. Build fresh
docker-compose build --no-cache
Important Change: Use: `docker-compose build airflow-build --no-cache`

6. Verify single image exists
docker images | Select-String "airflow"
- Should show ONE line: airflow-rag-custom:latest

7. Start services
docker-compose up -d

8. Wait 40 seconds, then check
Start-Sleep -Seconds 40
docker-compose ps

### Checks:
9. curl http://localhost:6333/health

10. List DAGs that Airflow sees: docker-compose exec airflow-webserver airflow dags list

11. docker-compose exec airflow-scheduler python -c "import sys; sys.path.insert(0, '/opt/airflow/plugins'); from rag_utils import load_and_enrich_data; print('RAG utils OK')"

12. docker-compose ps
