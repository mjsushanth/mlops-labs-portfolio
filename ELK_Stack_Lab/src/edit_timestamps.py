# edit_timestamps.py
import json
from datetime import datetime, timedelta

LOG_FILE = '../logstash/fraud_training.log'
SPACING_MINUTES = 30  # Space logs 30 minutes apart

def edit_log_timestamps():
    print("Reading existing logs...")
    
    # Read all logs
    with open(LOG_FILE, 'r') as f:
        logs = [json.loads(line) for line in f]
    
    print(f"Found {len(logs)} log entries")
    
    # Starting timestamp (5 days ago)
    start_time = datetime.now() - timedelta(days=5)
    
    # Update each log's timestamp
    for i, log in enumerate(logs):
        new_timestamp = start_time + timedelta(minutes=i * SPACING_MINUTES)
        log['timestamp'] = new_timestamp.isoformat()
        print(f"  Iteration {log['iteration']}: {new_timestamp}")
    
    # Write back to file
    print(f"\nWriting updated logs to {LOG_FILE}...")
    with open(LOG_FILE, 'w') as f:
        for log in logs:
            f.write(json.dumps(log) + '\n')
    
    print("Done! Logs now span 5 days with 30-minute intervals.")

if __name__ == "__main__":
    edit_log_timestamps()