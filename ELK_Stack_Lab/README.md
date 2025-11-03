
### MLops - Lab Submission.
#### Course IE 7374 - MLOps, Northeastern University.

#### Tools:
- Python 3.12, XGBoost, scikit-learn, pandas, Docker, Elasticsearch 7.16, Logstash 7.16, Kibana 7.16, UV package manager

#### Key Achievements Summary:
- MLOps pipeline integrating XGBoost fraud detection with ELK Stack monitoring infrastructure.
- Optimized **XGBoost model** achieving 95.3% recall using F2-score optimization and adaptive cost-based weighting
- Automated logging system generating structured JSON metrics logs from **25 training iterations** with natural variation
- **Configured Docker-based ELK stack** with **Elasticsearch, Logstash, and Kibana** using custom volume mounts.
- Designed Logstash pipeline to parse JSON logs, transform data types, and index into Elasticsearch time-series database
- Created **advanced Kibana dashboards** with multi-metric line charts, precision-recall trade-off visualizations, and confusion matrix tables - 5 visuals. Exported and saved as `export.ndjson`
- Hyperparameter impact heat map - correlating learning rates and tree depths with model F2 performance
- Validated complete data flow from model training through log ingestion to real-time dashboard visualization
- Containerization, persistent storage, monitoring, and performance tracking over time


1. ELK Stack = 3 Tools Working Together:
   - Logstash - Collects and processes data (like a data collector)
   - Elasticsearch - Stores and searches the data (like a smart database)
   - Kibana - Shows the data in charts and graphs (like a dashboard)

2. Local Installations:
   - Not necessary: Decided on using docker based approach.

3. Additional / Extensions:
   - There ARE Python client libraries that let Python talk to ELK:
   - pip install elasticsearch - Python client to interact with Elasticsearch server
   - pip install python-logstash - Python logging handler

4. High level idea from the prof's lab:
```
    Python Scripts → Generate Logs → Logstash → Elasticsearch → Kibana Dashboards
        ↓                              ↓            ↓              ↓
    train_model.py              Parse/Filter    Store/Index    Visualize
    drift_detection.py          Transform       Search         Charts
```
    - Gonna try attempting this stack on credit card fraud detection model + retraining pipeline.
  

5. Achieved Work on Model:
   - 2 notebooks, one for EDA and one for model experimentation and deep learning about XGBoost tuning.
   - Once achieved, the src/fraud_training.py script - finalized.
   - Added new particular elements for randomness in data, randomness in train-test split, and logging to Logstash. 
   - Multiple runs of the script generate different model metrics and log entries. As of now, 25 logs exist.
   - Proceeding, this work enables the ELK stack work with Logstash, Elasticsearch, and Kibana.
  
1. About logstash:
   - INPUT → FILTER → OUTPUT
   ```
   input {
   # Define data sources
   }

   filter {
   # Transform and enrich data
   }

   output {
   # Send data somewhere
   }
   ```
   - file { } - I'm reading from a file
   - path => "/logstash_dir/fraud_training.log"
   - "end" = only read NEW lines added after Logstash starts
   - /dev/null = forget everything, always read from beginning
   - codec => "json" - data format is JSON. parse each line as JSON.
   - type => "fraud_training" - tag the data with a type label
   - filters mutate the data - parse fields, convert types, add tags


1. Logs - Location: ELK_Stack_Lab/logstash/fraud_training.log.
2. Elasticsearch Data: Docker volume elastic_data; fraud-monitoring-* index. Searchable, queryable data.
   - LOST if: docker-compose down -v 
3. Kibana Dashboards & Visualizations: LOST if: docker-compose down -v
   - Elasticsearch documents in .kibana index
4. Best option - Export Dashboard. 

