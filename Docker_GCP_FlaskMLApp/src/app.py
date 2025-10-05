"""
app.py
------
Practicing Flask application for SEC Filings EDA API.

Flask Concepts Demonstrated:
1. App initialization with Flask(__name__)
2. Route decorators (@app.route)
3. JSON responses with jsonify()
4. Error handling with try/except
5. Development server with app.run()

    python src/app.py
    
Then visit:
    http://localhost:5000/
    http://localhost:5000/stats
    http://localhost:5000/benchmark
    http://localhost:5000/outliers  
app.py (main Flask app)
    ↓
Registers routes (@app.route decorators)
    ↓
Routes call service functions
    ↓
Services return JSON data
    ↓
Flask sends JSON response to client
"""

from flask import Flask, jsonify, request
from datetime import datetime
import traceback

# adding src so that docker wont get issues. 
# Gunicorn runs src.app:app, meaning it treats src as a package. All imports must be relative to project root.
from src.stats_service import get_overall_stats
from src.benchmark_service import run_benchmark
from src.outlier_service import detect_text_outliers
from src.data_loader import get_pandas_data, get_polars_data



# ========================================
# FLASK APP INITIALIZATION
# ========================================

# __name__ helps Flask locate templates/static files
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False  
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True  

print("\n" + "=" * 60)
print("FLASK APP STARTING - PRE-LOADING DATA")
print("=" * 60)

# called ONCE when Flask starts, before handling any requests
df_pandas = get_pandas_data()
df_polars = get_polars_data()

print(f"Pandas DataFrame loaded: {df_pandas.shape}")
print(f"Polars DataFrame loaded: {df_polars.shape}")
print("=" * 60 + "\n")


# ========================================
# ROUTE 1: ROOT/HOME ENDPOINT
# ========================================

@app.route('/')
def home():
    """
    Root endpoint - API documentation and health check.
    
    Flask Concepts:
    - @app.route('/') decorator registers this function for the root URL
    - jsonify() converts Python dict to JSON response
    - Returns JSON with 200 OK status by default

        curl http://localhost:5000/
        OR visit http://localhost:5000/ in browser
    """

    return jsonify({
        "service": "SEC Filings EDA API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/": "This help message",
            "/stats": "Overall dataset statistics (Polars aggregations)",
            "/benchmark": "Pandas vs Polars performance comparison",
                "/outliers": "Detect anomalous sentences using ML",  
        },
        "dataset": {
            "rows": len(df_pandas), "columns": len(df_pandas.columns)
        }
    })



# ROUTE 2: STATISTICS ENDPOINT

@app.route('/stats')
def stats():
    """
    Return overall dataset statistics using Polars.
    - route decorator, calls external service, error handling with try/except
    - 500 status code on error
        curl http://localhost:5000/stats
    """
    try:
        stats_data = get_overall_stats(df_polars)
        
        return jsonify({
            "endpoint": "/stats",
            "timestamp": datetime.now().isoformat(),
            "data": stats_data
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ROUTE 3: BENCHMARK ENDPOINT

@app.route('/benchmark')
def benchmark():
    """
    Compare Pandas vs Polars performance on same operations.    
    - Multiple routes can exist in same file. Each route is independent
        curl http://localhost:5000/benchmark
    """
    try:
        # Call benchmark service (we'll write this next)
        benchmark_results = run_benchmark(df_pandas, df_polars)
        
        return jsonify({
            "endpoint": "/benchmark",
            "timestamp": datetime.now().isoformat(),
            "results": benchmark_results
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500



# ROUTE 4: OUTLIER DETECTION ENDPOINT

# ========================================
# ROUTE 4: OUTLIER DETECTION ENDPOINT
# ========================================

@app.route('/outliers')
def outliers():
    """
    Detect anomalous sentences using Isolation Forest.
    
    Flask Concepts:
    - Same pattern as other routes
    - This one takes longer (ML model training)
    - Uses Polars for feature extraction
    
    Try it:
        curl http://localhost:5000/outliers
    """
    try:
        # Use quick summary for faster response
        # For full analysis, use: detect_text_outliers(df_polars)
        outlier_results = detect_text_outliers(df_polars, contamination=0.05)  
        
        return jsonify({
            "endpoint": "/outliers",
            "timestamp": datetime.now().isoformat(),
            "results": outlier_results
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500



# ERROR HANDLERS (Optional)

@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 Not Found errors.
    
    Flask Concepts:
    - @app.errorhandler(code) catches specific HTTP errors
    - Returns custom JSON instead of HTML error page
    """
    return jsonify({
        "error": "Endpoint not found",
        "status": 404,
        "available_endpoints": ["/", "/stats", "/benchmark"]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 Internal Server errors."""
    return jsonify({
        "error": "Internal server error",
        "status": 500,
        "message": str(error)
    }), 500




# ========================================
# RUN FLASK DEVELOPMENT SERVER


if __name__ == '__main__':
    """
    Run Flask development server.
    - debug=True enables auto-reload on code changes
    - host='0.0.0.0' makes server accessible from other machines
    - port=5000 is Flask's default port
    
    WARNING: Never use debug=True in production!
    For production, use: gunicorn app:app
    """
    print("\n Starting Flask development server...")
    print("API Documentation: http://localhost:5000/")
    print("Statistics: http://localhost:5000/stats")
    print("Benchmark: http://localhost:5000/benchmark")
    print("Outlier Detection: http://localhost:5000/outliers")
    print("\nPress CTRL+C to stop\n")
    
    app.run(
        debug=True,      # Auto-reload on code changes
        host='0.0.0.0',  # Listen on all network interfaces
        port=5000        # Port number
    )


# ========================================

# Docker related:
## Bytecode is platform-specific: copy Windows .pyc files into a Linux container, they're useless or cause errors. Python will regenerate them anyway.
## Source code is enough: Python automatically compiles .py → .pyc on first run inside the container. 
## Reproducibility: Fresh compilation in the container ensures consistent behavior.
## Let the destination factory assemble it fresh (pip install). 

"""
The Dockerfile contains:
    FROM - base image
    RUN - commands to execute during build
    COPY - files to include
    CMD - what to run when container starts

These commands are run by YOU in the terminal, not written in the Dockerfile:

Command           Purpose                          When You Use It
docker build      Create image from Dockerfile    Once per code change
docker run        Start container from image      To launch your app
docker ps         List running containers          Check what's running
docker logs       View container output            Debug issues
docker stop       Stop container                  Shut down app
docker images     List images                     See what's built
docker prune      Clean up unused images/containers Free disk space
docker-compose    Manage multi-container apps     Not needed for single Flask app

Dockerfile: Static recipe (build instructions)
Docker CLI: Dynamic actions (build, run, manage)
docker-compose: Optional orchestration (multi-service)
"""