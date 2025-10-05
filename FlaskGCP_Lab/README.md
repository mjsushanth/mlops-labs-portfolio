# Flask GCP + Docker Practice, 
# Includes practice with Polars and PyOD for anomaly detection ( IsolationForest algorithm )

This is a sample project demonstrating a Flask web application deployed on Google Cloud Platform (GCP) using Docker. The application processes SEC filings data, calculates statistics using Polars, and performs anomaly detection with PyOD.

SEC filings data is the dataset chosen for my main course project, so i've converted it into a parquet format. Now, this lab project focuses helping with exploring the data, calculating statistics, and performing anomaly detection using PyOD. 

The **SCREENSHOTS** present in the `results` folder show the web app's interface and outputs. Flask has been successfully developed, and the app is tested. 

The application is containerized with Docker and can be deployed on GCP App Engine or Cloud Run.

## Main Features:
**Data Engineering** - Loaded/cached 200k SEC filings (HuggingFace → Parquet)
**EDA Service** - Polars-powered statistics endpoint
**Benchmarking** - Pandas vs Polars performance comparison (2.5x speedup)
**ML Component** - Isolation Forest anomaly detection (PyOD)
**API Layer** - Flask with 4 working endpoints
**Containerization** - Docker image (1.5GB), multi-stage build
**Production Server** - Gunicorn with 2 workers
    `Browser → localhost:8080 → Docker container → Gunicorn → Flask app`

**Local Testing** - All endpoints verified working
**GCP Deployment** - Cloud Run deployment with public URL.

**Success Items**:
- Flask app with multiple routes, Tested locally - Successful.
- Polars for fast data processing, Tested locally - Successful.
- PyOD for anomaly detection, Tested locally - Successful.
- Dockerfile for containerization, Built and tested - Successful. ( Screenshots in `results` folder )
- GCP deployment (App Engine or Cloud Run), Deployed and tested - Successful.


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


## Google cloud:

- Cloud Run: Free (under 2M requests/month)
- Artifact Registry: $0.10/GB/month (1.5GB image = $0.15/month)
- Cloud Build: 120 free builds/day
- Total monthly cost: ~$0.15 (fifteen cents) ???
