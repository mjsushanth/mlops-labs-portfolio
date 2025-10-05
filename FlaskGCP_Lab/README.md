# Flask GCP + Docker Practice, 
# Includes practice with Polars and PyOD for anomaly detection ( IsolationForest algorithm )

This is a sample project demonstrating a Flask web application deployed on Google Cloud Platform (GCP) using Docker. The application processes SEC filings data, calculates statistics using Polars, and performs anomaly detection with PyOD.

SEC filings data is the dataset chosen for my main course project, so i've converted it into a parquet format. Now, this lab project focuses helping with exploring the data, calculating statistics, and performing anomaly detection using PyOD. The **SCREENSHOTS** present in the `results` folder show the web app's interface and outputs. Flask has been successfully developed, and the app is tested. The application is containerized with Docker and can be deployed on GCP App Engine or Cloud Run.

## Main Features:
1. **Data Engineering** - Loaded/cached 200k SEC filings (HuggingFace → Parquet)
2. **EDA Service** - Polars-powered statistics endpoint
3. **Benchmarking** - Pandas vs Polars performance comparison (2.5x speedup)
4. **ML Component** - Isolation Forest anomaly detection (PyOD)
5. **API Layer** - Flask with 4 working endpoints
6. **Containerization** - Docker image (1.5GB), multi-stage build
7. **Production Server** - Gunicorn with 2 workers
    `Browser → localhost:8080 → Docker container → Gunicorn → Flask app`
8. **Local Testing** - All endpoints verified working
9. **GCP Deployment** - Cloud Run deployment with public URL. Docker - GCP workflow practiced.

**Success Items**:
- **Flask app with multiple routes**, Tested locally - Successful.
- **Polars** for fast data processing, Tested locally - Successful.
- **PyOD** for anomaly detection, Tested locally - Successful.
- **Dockerfile** for containerization, Built and tested - Successful. ( Screenshots in `results` folder )
- **GCP deployment** (App Engine or Cloud Run), Deployed and tested - Successful. ( Screenshots in `results` folder )


# Flask Strategy, and Endpoints:
- Flask app with 4 endpoints: `/`, `/stats`, `/benchmark`, `/outliers`.
- `/` - Home route with welcome message and instructions.
- `/stats` - Returns basic statistics (mean, median, std, min, max) for numeric columns using Polars.
- `/benchmark` - Compares performance of Pandas vs Polars for calculating statistics on the dataset.
- `/outliers` - Uses PyOD's Isolation Forest to detect anomalies in numeric columns and returns count of outliers.
- Each endpoint handles errors and returns JSON responses.
- Has singleton pattern, lazy loading of dataset for efficiency, and caching of results to speed up repeated requests.


# Docker Strategy:
- Multi-stage build to minimize final image size (1.5GB).
- No conda image, Conda bulks and makes 3-5 GB or more. Used **slim python** base image.
- Install only necessary packages via pip (Polars, PyOD, Flask, Gunicorn).
- Copy only source code and requirements.txt into the image.
- Ignore unnecessary files with .dockerignore. Results, datasets, .git, __pycache__, .vscode etc.
- Use Gunicorn as production server with 2 workers for handling requests.
- Expose port 8080 for Cloud Run compatibility.
- Set memory to 2GB in Cloud Run for handling 200k rows efficiently.

# Instructions to Run Locally for Docker Test:
1. Ensure Docker is installed and running on your machine.
    docker build -t flaskgcplab:latest .
    docker run -p 8080:8080 flaskgcplab:latest

- Build image locally with tag flaskgcplab:latest. Image stored in local Docker registry (on your machine)
- Run container locally


# Gcloud:

- First, install Google Cloud SDK: https://cloud.google.com/sdk/docs/install - this has been installed on my Windows machine.
- Authenticate with your Google account: `gcloud auth login`
- Set your project: `gcloud config set project YOUR_PROJECT_ID`
- Enable necessary APIs: 
    - For cloud run, artifact registry, and cloud build, enable these APIs. 
    - ( 1 hosts docker images auto-scales and gives public URL, 2 stores Docker image, 3 builds them )
        1. run.googleapis.com
        2. artifactregistry.googleapis.com
        3. cloudbuild.googleapis.com

## High level flow after installing gcloud sdk:

1. Tag your local Docker image for GCP
2. Push image to Artifact Registry
3. Deploy to Cloud Run
4. Get public HTTPS URL

# *Step 1*: Create Artifact Registry Repository
    - gcloud artifacts repositories create flask-repo --repository-format=docker --location=us-central1 --description="Flask EDA app images"
    - Creates a Docker repository named `flask-repo` in `us-central1` region.

# *Step 2*: Configure Docker to Authenticate with GCP
    - gcloud auth configure-docker us-central1-docker.pkg.dev 
    - Sets up Docker to use GCP credentials for pushing images.

# *Step 3*: Tag Your Image
    - docker tag flaskgcplab:latest us-central1-docker.pkg.dev/sec10k-flask-docker/flask-repo/flaskgcplab:latest
    - Tags local image `flaskgcplab:latest` for GCP Artifact Registry.

# *Step 4*: Push to GCP (Takes 5-10 Minutes)
    - docker push us-central1-docker.pkg.dev/sec10k-flask-docker/flask-repo/flaskgcplab:latest
    - Uploads the tagged image to Artifact Registry.

# *Step 5*: Cloud Run
    - gcloud run deploy flask-eda-service --image us-central1-docker.pkg.dev/sec10k-flask-docker/flask-repo/flaskgcplab:latest --platform managed --region us-central1 --allow-unauthenticated --port 8080 --memory 2Gi

    - Cloud Run creates a service named flask-eda-service
    - Pulls your image from Artifact Registry
    - Starts container listening on port 8080
    - Allocates 2GB memory (your app needs this for 200k rows)

# Links:
    https://flask-eda-service-245922041291.us-central1.run.app/
    https://flask-eda-service-245922041291.us-central1.run.app/stats
    https://flask-eda-service-245922041291.us-central1.run.app/benchmark
    https://flask-eda-service-245922041291.us-central1.run.app/outliers

    Got public HTTPS URL (globally accessible)



# When re-developments are done:
    - Rebuild local Docker image: docker build -t flaskgcplab .
    - Tag it again for GCP: docker tag flaskgcplab:latest us-central1-docker.pkg.dev/sec10k-flask-docker/flask-repo/flaskgcplab:latest
    - Push updated image: docker push us-central1-docker.pkg.dev/sec10k-flask-docker/flask-repo/flaskgcplab:latest
    - Redeploy to Cloud Run (same command as above): gcloud run deploy flask-eda-service --image us-central1-docker.pkg.dev/sec10k-flask-docker/flask-repo/flaskgcplab:latest --platform managed --region us-central1 --allow-unauthenticated --port 8080 --memory 2Gi

    - **Commands**: 
    docker build -t flaskgcplab:latest .
    docker tag flaskgcplab:latest us-central1-docker.pkg.dev/sec10k-flask-docker/flask-repo/flaskgcplab:latest
    docker push us-central1-docker.pkg.dev/sec10k-flask-docker/flask-repo/flaskgcplab:latest
    gcloud run deploy flask-eda-service --image us-central1-docker.pkg.dev/sec10k-flask-docker/flask-repo/flaskgcplab:latest --region us-central1



# WHEN DONE; to delete and save costs:
    - gcloud run services delete flask-eda-service --region us-central1
    - gcloud artifacts repositories delete flask-repo --location=us-central1

**Alternate Methods**:
    - Console → Artifact Registry → Create Repository (click through forms)
    - Console → Cloud Run → Create Service etc. 



# Student Readup For Docker: 

## Docker related:
- Bytecode is platform-specific: copy Windows .pyc files into a Linux container, they're useless or cause errors. Python will regenerate them anyway.
- Source code is enough: Python automatically compiles .py → .pyc on first run inside the container. 
- Reproducibility: Fresh compilation in the container ensures consistent behavior.
- Let the destination factory assemble it fresh (pip install). 


# The Dockerfile contains:
    FROM - base image
    RUN - commands to execute during build
    COPY - files to include
    CMD - what to run when container starts

These commands are run by YOU in the terminal, not written in the Dockerfile:

`Command`           `Purpose`                             `When You Use It`
docker build      Create image from Dockerfile          Once per code change
docker run        Start container from image            To launch your app
docker ps         List running containers               Check what's running
docker logs       View container output                 Debug issues
docker stop       Stop container                        Shut down app
docker images     List images                           See what's built
docker prune      Clean up unused images/containers     Free disk space
docker-compose    Manage multi-container apps           Not needed for single Flask app

`Dockerfile:` Static recipe (build instructions)
`Docker CLI:` Dynamic actions (build, run, manage)
`docker-compose:` Optional orchestration (multi-service)

## Docker's config.json to use gcloud credentials:
    gcloud auth configure-docker us-central1-docker.pkg.dev

    - This command earlier? it modifies Docker's config.json to use gcloud credentials for authentication when pushing/pulling images to/from GCP Artifact Registry.

## Google cloud:

- Cloud Run: Free (under 2M requests/month)
- Artifact Registry: $0.10/GB/month (1.5GB image = $0.15/month)
- Cloud Build: 120 free builds/day
- Total monthly cost: ~$0.15 (fifteen cents) ???
