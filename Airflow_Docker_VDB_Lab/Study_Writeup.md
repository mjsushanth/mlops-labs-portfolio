## Write up for learning and breakdowns:

```
My Machine
├── Local conda env (testing/development)
└── Docker containers (actual execution)
    ├── Airflow container (has its own Python)
    ├── Postgres container 
    └── Qdrant container
```

### Steps to create local environment for testing/development:

- mamba env create -f env/airflow_qdrant_lab.yml
- conda activate airflow_qdrant_lab
- python --version  # Should show 3.11.x

### Recap / Knowledge write-up:

- What is .dockerignore? .dockerignore is your "DO NOT PACK" list. When Docker builds an image, it first copies your entire directory into the "build context". This file says "skip these items."
- __pycache__/: Python creates these. They're machine-specific, would break in container
- env/: conda environment. Would bloat image massively
- .git/: Git history can be huge, container doesn't need it
- notebooks/: Development artifacts, not for production
- .DS_Store: MacOS metadata, .vscode/: editor settings - personal, not portable



### Multi container architecture recap:
1. For older labs, practices, we did: (look at FlaskGCP Docker Lab.)
```
Local Docker → GCP Registry → Cloud Run (1 container)
```

2. Here, we do:
```
    5 containers (Postgres, Qdrant, Scheduler, Webserver, Init etc..)
    ↓
    Need orchestration tool
    ↓
    Docker Compose handles everything
```



### Direct - Transitive dependencies:
```
Your App
├── Direct dependencies (what you import)
│   ├── pandas (you use this)
│   ├── sentence-transformers (you use this)
│   └── qdrant-client (you use this)
└── Transitive dependencies (what they need)
    ├── numpy (pandas needs this)
    ├── torch (sentence-transformers needs this)
    └── grpcio (qdrant-client needs this)
```

### Docker quick recap:
```
Base (FROM) -> Admin Mode, Sys Changes (USER ROOT) -> Install system tools (RUN apt) 
    -> Switch back to normal user (USER AIRFLOW) 
        -> Install Python tools (RUN pip install) -> Copy your code (COPY) -> Set working directory (WORKDIR) etc.
```


### Docker Compose + Airflow recap:
- docker-compose is a tool that manages multiple Docker containers
    - Builds images (using Dockerfile)
    - Creates network for containers to talk
    - Starts all containers in correct order
    - Airflow starts INSIDE its containers
- In this implementation and practice, **Airflow runs entirely inside Docker containers**. 

```
Docker Compose (Conductor)
    ├── Reads docker-compose.yml 
    ├── Builds images from Dockerfile
    └── Starts containers
        ├── Postgres (db)
        ├── Qdrant (vector storage)
        ├── Airflow Scheduler (conductor - orchestrates tasks)
        └── Airflow Webserver (UI  - provides interface)
```


### Astronomer/Industry Standard:
- Astronomer is a managed Airflow service. They provide a standard Docker image for Airflow that includes best practices and optimizations.
- Using official Airflow image (industry standard)
- LocalExecutor for simplicity (CeleryExecutor for production)
- Proper service separation (database, app, vector store)
- Environment variables for configuration

### Compose yml breakdown:
1. Template - x-airflow-common. 
    - &airflow-common anchor means we define this once and reuse it with <<: *airflow-common later.
2. Database service - Postgres. Airflow needs a database to store metadata, task states, etc.
    - to store DAG history, task status, user info.
3. Qdrant service - Vector DB for storing embeddings.
    - Port mapping: "host:container" - Access Qdrant at localhost:6333 from your machine.
4. Airflow services - Scheduler and Webserver.
    - Inherits common settings from x-airflow-common. Access web UI at localhost:8080.
5. Volumes - Persist data outside containers. (optional, possibly)
    - postgres_data: keeps Postgres data across restarts.
    - qdrant_data: keeps Qdrant data across restarts.
    - airflow_logs: keeps Airflow logs outside container.

### Volume mounting recap:
- Volumes are like external hard drives for containers. 
- Don't use in Docker. Docker COPY would embed this into the image permanently.
- Instead, we use volume mounting in docker-compose.yml: ./data:/opt/airflow/data.
- Volume mounting means the file stays on your host, containers just get access to it.

### Requirements.txt - Dependencies 
- Two approaches: 
    - Dependencies installed DURING image build, Creates a self-contained image, Image is portable.
    - Dependencies installed EVERY container start, Slower startup.

```
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
```
- Approach A because it's the production standard - build once, run many times.


### Docker commands - Previous vs Current approach:

```
# 1. BUILD - Create image from Dockerfile
    docker build -t flaskgcplab:latest .

# 2. TAG - Rename for registry
    docker tag flaskgcplab:latest us-central1-docker.pkg.dev/.../flaskgcplab:latest

# 3. PUSH - Upload to registry
    docker push us-central1-docker.pkg.dev/.../flaskgcplab:latest

# 4. RUN - Start single container
    docker run -p 8080:8080 flaskgcplab:latest
```

#### Everything in one command!
`docker-compose up`

Behind the scenes, docker-compose does:
1. BUILD all images (reads Dockerfile)
2. CREATE network for containers
3. START containers in dependency order
4. MOUNT volumes
5. SET environment variables

When you run docker-compose up, it actually runs: **What docker-compose does internally:**
- docker build -t airflow_docker_vdb_lab_airflow-webserver .
- docker build -t airflow_docker_vdb_lab_airflow-scheduler .
- docker network create airflow_docker_vdb_lab_default
- docker volume create airflow_docker_vdb_lab_postgres_data
- docker run --name postgres --network airflow_docker_vdb_lab_default postgres:13
- docker run --name qdrant --network airflow_docker_vdb_lab_default qdrant/qdrant
- ... and more


### Docker Importance:
- Traditional: Write Python → Test locally → Dockerize → Discover issues → Rewrite
- Modern: Setup Infrastructure → Verify services → Write code in actual environment
- **ML, Local Dev first** versus **Data Infrastructure first** approach.

### Practical Observations on Docker initialization and build work:
- pinned direct dependencies, but not transitive ones. 
- airflow image already has some packages, and our request adds pinned items. thus, dependancy resolution happens.
- took almost 10+ minutes. grpcio versions fighting with each other. 
- pip is quite slow: sequential installer, downloads-fails-backtracks, multiple solve attempts. 


### Understanding data mounting further:
#### Mounted = Shared Folder.
- Symbolic link between host and container. Like pointer. `string* mount = &original_file;`
- Windows: Uses SMB/CIFS protocol
- Linux containers: Use bind mounts

```
Your Windows PC                     Docker Container
D:\...\data\exports\       ←→      /opt/airflow/data/exports/
    ↑                                      ↑
    Your actual file                 Same file, different path
```

```
Your Parquet File (Data)          PostgreSQL (Metadata)
├── SEC filings text       vs     ├── DAG run history
├── Company info                  ├── Task success/failure logs
└── 200k sentences                ├── User accounts (admin)
                                  ├── Variables/connections
                                  └── Schedule information
```

```
SERVICE          PURPOSE                         STORES WHAT?
─────────────────────────────────────────────────────────────
PostgreSQL    →  Airflow's memory/brain      →  Metadata only
Qdrant        →  Vector database             →  Embeddings (will store)
Airflow       →  Orchestrator/scheduler      →  Nothing (uses Postgres)
Data Parquet  →  Actual data                 →  On YOUR disk (mounted)
```

```
1. Your parquet on Windows disk
        ↓ (mounted/shared)
2. Airflow reads it
        ↓ (processes)
3. Creates embeddings
        ↓ (stores)
4. Qdrant holds vectors
```


### WHERE is my stuff?:

Windows PC
│
├── D:\...\Airflow_Docker_VDB_Lab\  (mini-lab folder)
│   │
│   ├── 📁 qdrant_storage\         ←  500MB EMBEDDINGS LIVE HERE!
│   │   └── collections\
│   │       └── sec_filings\
│   │           └── [vector data files]
│   │
│   ├── 📁 postgres_data\          ← POSTGRES DATABASE FILES HERE!
│   │   └── pg_data\
│   │       └── [airflow tables]
│   │
│   ├── 📁 dags\                   ←  Python code (visible)
│   ├── 📁 plugins\                ←  Python code (visible)
│   └── 📁 data\                   ←  parquet (visible)

- Docker Volumes = Real Windows Folders.
- embeddings are REAL FILES on your Windows disk.



### For local Mamba:

    conda deactivate
    mamba env remove -n airflow_qdrant_lab
    mamba env create -f env/airflow_qdrant_lab.yml
    conda activate airflow_qdrant_lab
    python -m ipykernel install --user --name airflow_qdrant_lab --display-name "Airflow Qdrant Lab"


### Airflow Related:
- What's Happening with **context: In Airflow, every task function receives a context dictionary with metadata about the DAG run:
    - task_instance - the current task object
    - execution_date - when this run started
    - dag_run - the overall pipeline run
    - The **context unpacks this dictionary. You use it to:
- Pass data between tasks using XCom (cross-communication)
- Access run metadata

- **Temp files or DB**: In production, use a proper database (Postgres, MySQL). Temp files are for local testing only.
- **XCom**: Airflow's way to pass small data between tasks.
- Airflow tasks run in separate Python processes - they can't share memory directly. Options:
    - XCom - Limited to 48KB, too small for dataframes
    - Database - Overkill for temporary data
    - Files - Simple, works for any size

```
Task 1 Process        Task 2 Process
    |                      |
    v                      v
Write to /tmp  -->  Read from /tmp
```


### THIS - is bad. This is the bloating issue:
- Docker Compose creates separate image tags even though they're identical layers underneath. It's poor design that wastes disk space.
```
airflow-webserver:
  build: .      # ← Builds image, tags as airflow-webserver:latest
  
airflow-scheduler:
  build: .      # ← Builds AGAIN, tags as airflow-scheduler:latest
  
airflow-init:
  build: .      # ← Builds AGAIN, tags as airflow-init:latest
```

- Docker Compose's YAML anchors (&airflow-common) are powerful but dangerous. When you put build: in an anchor and reuse it 3+ times, Docker's build context gets confused about whether these are:
    - The same image (should share layers)
    - Different images (might diverge)
    - Build-time vs run-time concerns
- cache thrashing, redundant builds, and the 3x10GB nightmare !!