## Airflow + Docker + Qdrant Practice Lab.

This lab aims to practice, present the format and learnings for how to orchestrate **multiple services** using **Docker Compose**, specifically for an **Airflow setup** with a **Postgres database and Qdrant vector database**. This lab requires being a little familiar with Docker, Docker Compose.


### Lab Success / Features:
- **Docker - Qdrant + Airflow + Postgres** orchestration (5-task DAG)
- **Docker multi-service setup** (Postgres, Qdrant, Airflow)
- **RAG pipeline** (data → chunks → embeddings → vector DB)! Smart chunking, load slicing, windowing etc.
- **Embeddings** generated with MPNet model. 
- **Debugging Docker** - Image Bloating due to torch/nvidia, cache issue, triple-build issues, mismatched tags issues etc.
- **log forensics** - Practiced a lot of log forensics with shell + docker commands. `Study_Writeup.md` has all learning recorded and explained, to be a useful material for anyone.
- **Optimized docker-compose** Attempted production level - custom image, no cache, profiles.
    - This was done several times, more than 15 - 20 times to get it right.
- **Parquet creation, IPC, Stream - NonStream Shards** - Investigated shards, pyarrow usage, JSONL usage, and multiple codes to achieve 28.5x compression. 44GB files compressed to 1.5 GB single parquet file with **71 million rows** !!


### Steps:
1. Please go through the `Study_Writeup.md` first, it explains the concepts and breakdowns in detail. 
2. To run everything, first the checklist is as follows:
    - Install Docker Desktop, Install Google Cloud SDK (if planning to push to GCP registry)
    - Ensure data files are in place: `data/`, or `data/exports/..` 
    - Ensure project has a proper structure; such as `dags/` for the Airflow DAGs, `notebooks/` for experimentation, `data/` for datasets, and `plugins/` for utils etc.
    - Ensure `docker-compose.yml`, `Dockerfile`, `.dockerignore`, `requirements.txt` are in the root folder.
3. docker-compose should use a standard image. its design is: version, x-airflow-common, services (db, qdrant, scheduler, webserver), volumes (optional).
4. We can still have local conda/mamba environment for development/testing.

### Steps start here:
5. We verify first if docker is running: `docker --version` and `docker-compose --version`.
6. Initialize the Airflow DB: `docker-compose up airflow-init` (runs once).
    - This creates database schema and admin user. 
    - WAIT for this specific message: airflow-init_1 exited with code 0. 
    ```
        airflow-init-1  | User "admin" created with role "Admin"
        airflow-init-1 exited with code 0
    ```

7. Build images: `docker-compose build` (Reads `Dockerfile`, Installs `requirements.txt` packages, Creates custom Airflow image).
    - Use command to check: `docker images` shows custom airflow image. 
    - **Important Change**: use `docker-compose build airflow-build --no-cache`.
    - Reason: profiles: [donotstart] hides the build service from Docker Compose's default commands.
    - We need clean solution: image reference is spelled out, **same image (should share layers)**, should avoid **cache thrashing, redundant builds, and the 3x10GB** nightmare bloating.
 

8. Start all services: `docker-compose up` (or `docker-compose up -d` detached mode (background)).
    - Creating network "airflow_docker_vdb_lab_default", similar messages for postgres, qdrant, scheduler, webserver, etc.
9. Verify Services are Healthy: `docker-compose ps` (Should show all services as "Up").
```
    Expected output (all should be "running"):
    NAME                                    STATUS              PORTS
    airflow_docker_vdb_lab_postgres_1      running             5432/tcp
    airflow_docker_vdb_lab_qdrant_1        running             0.0.0.0:6333->6333/tcp
    airflow_docker_vdb_lab_scheduler_1     running             
    airflow_docker_vdb_lab_webserver_1     running             0.0.0.0:8080->8080/tcp
```

```
    airflow-rag-custom   latest    8c2db79f08ce   10 minutes ago   4.05GB
```

#### Extra commands:
10. Check logs. `docker-compose logs` and `docker-compose logs -f` (follow mode).
11. Test services:
    - Airflow UI: http://localhost:8080 (login with admin/admin)
    - Qdrant UI: http://localhost:6333/dashboard (no auth by default)
    - Postgres: Use a client like DBeaver or pgAdmin to connect to.
12. Default creds:
    - Airflow: admin / admin (change after first login)
    - Postgres: airflow / airflow (user/pass as per docker-compose.yml)
    - Qdrant: no auth by default
13. Verify accesibility. `docker-compose exec airflow-webserver ls -la /opt/airflow/data/exports/` (should show files in exports/ folder).
14. Stop services: `docker-compose down` or `docker-compose down -v` (stops and removes containers, network, but preserves volumes).
    - down    = Turn off computer (keeps hard drive)
    - down -v = Turn off computer + format hard drive


```Example log on checking just the webserver status:
(airflow_qdrant_lab) PS D:....> docker-compose ps airflow-webserver
NAME                                         IMAGE                                      COMMAND                  SERVICE             CREATED         STATUS                   PORTS
airflow_docker_vdb_lab-airflow-webserver-1   airflow_docker_vdb_lab-airflow-webserver   "/usr/bin/dumb-init …"   airflow-webserver   2 minutes ago   Up 2 minutes (healthy)   0.0.0.0:8080->8080/tcp, [::]:8080->8080/tcp
```


### Complete Picture
```
graph TD
    A[docker-compose up] --> B[Read docker-compose.yml]
    B --> C[Build from Dockerfile]
    B --> D[Pull pre-built images]
    C --> E[Create Network]
    D --> E
    E --> F[Start Postgres]
    F --> G[Start Qdrant]
    G --> H[Run airflow-init]
    H --> I[Start Scheduler]
    I --> J[Start Webserver]
    J --> K[Mount volumes - ./dags, ./data]
    K --> L[Ready for Python development]
```


### Technical readme on what's done:
1. I created a custom parquet format by merging shards (did it both on small_full and large_full).
2. Using small_full, 200k rows, ~40MB. 
3. Created a DAG that reads the parquet, creates embeddings, and upserts to Qdrant.
   - Idea: t1_extract >> t2_chunk >> t3_embed >> t4_upsert >> t5_test
5. NLP like ideas: data filtering, window size, chunk creation, generate_embeddings, and talk to Qdrant.
6. Used polars for speed (faster than pandas, as proven in Flask lab).
7. Using sentence-transformers/all-MiniLM-L6-v2 model (small, fast, good quality).


### Tech-Related, for the ML algo part. 
### Ongoing: (Scaling reality of embeddings using - MPNet model, not miniLM)
- Small_full: 200k sentences → 95k chunks → 4.5 hours
- 730/5937: Processed 730 batches out of 5937 total batches
- 12%: Only 12% complete
- 35:06: 35 minutes and 6 seconds elapsed so far
- Large_full: 71.8M sentences (359x larger) → ~34M chunks → 1,600+ hours (67 days!)
- Wont scale.

