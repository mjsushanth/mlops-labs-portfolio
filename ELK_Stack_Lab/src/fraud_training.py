# fraud_training.py
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, fbeta_score
)
import xgboost as xgb

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_PATH = '../dataset/creditcard.csv'
LOG_FILE = '../logstash/fraud_training.log'
MIN_RECALL = 0.95

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_next_iteration():
    """Read log file and return next iteration number"""
    if not os.path.exists(LOG_FILE):
        return 1
    
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            if not lines:
                return 1
            # Read last line, get iteration number
            last_entry = json.loads(lines[-1])
            return last_entry.get('iteration', 0) + 1
    except:
        return 1

def add_realistic_noise(X, y, noise_level=0.02):
    """Add small noise to simulate data variation between runs"""
    X_noisy = X.copy()
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X_noisy + noise
    return X_noisy, y

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_fraud_model():
    
    iteration = get_next_iteration()
    print(f"\n{'='*70}")
    print(f"Fraud Detection Training - Iteration {iteration}".center(70))
    print(f"{'='*70}\n")
    
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Loaded {len(df):,} transactions\n")
    
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    # Split with random state. 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        stratify=y  
    )
    
    # small noise for variation
    X_train_noisy, y_train_noisy = add_realistic_noise(X_train, y_train)
    
    print(f"Data split:")
    print(f"  Train: {len(X_train):,} samples ({y_train.sum()} frauds)")
    print(f"  Test:  {len(X_test):,} samples ({y_test.sum()} frauds)\n")
    
    # Calculate adaptive weight
    standard_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    avg_fraud_amount = df[df['Class']==1]['Amount'].mean()
    cost_missing_fraud = avg_fraud_amount
    cost_false_alarm = 25
    cost_ratio = cost_missing_fraud / cost_false_alarm
    
    adjustment_factor = min(np.sqrt(cost_ratio), 3.0)
    adaptive_weight = standard_weight * adjustment_factor
    
    print(f"Scale pos weight: {adaptive_weight:.1f}\n")
    
    # Train model with slight hyperparameter variation
    params = {
        'max_depth': np.random.randint(4, 7),  # 4, 5, or 6
        'learning_rate': np.random.uniform(0.03, 0.07),  # Small variation
        'n_estimators': np.random.randint(80, 120),  # 80-120 trees
        'scale_pos_weight': adaptive_weight,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False
    }
    
    print("Training XGBoost...")
    print(f"  Hyperparameters: depth={params['max_depth']}, "
          f"lr={params['learning_rate']:.4f}, n_est={params['n_estimators']}")
    
    start_time = datetime.now()
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_noisy, y_train_noisy, verbose=False)
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"  Training completed in {training_time:.2f}s\n")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.85, 0.875, 0.9]
    best_f2 = 0
    best_threshold = 0.5
    best_metrics = None
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
        
        # Check recall constraint
        if recall >= MIN_RECALL and f2 > best_f2:
            best_f2 = f2
            best_threshold = threshold
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'f2': f2,
                'y_pred': y_pred
            }
    
    # If no threshold meets recall requirement, pick best F2
    if best_metrics is None:
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
            
            if f2 > best_f2:
                best_f2 = f2
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'f2': f2,
                    'y_pred': y_pred
                }
    
    # Calculate final metrics
    y_pred_final = best_metrics['y_pred']
    
    accuracy = accuracy_score(y_test, y_pred_final)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Print results
    print("Results:")
    print(f"  Threshold: {best_threshold:.2f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    print(f"  F1 Score:  {best_metrics['f1']:.4f}")
    print(f"  F2 Score:  {best_metrics['f2']:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Frauds caught: {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)")
    print(f"  False alarms:  {fp}\n")
    
    # Prepare log entry
    metrics_to_log = {
        'timestamp': datetime.now().isoformat(),
        'iteration': iteration,
        'model': 'XGBoost',
        'optimization_strategy': 'F2_with_recall_constraint',
        
        # Core metrics
        'accuracy': round(accuracy, 4),
        'precision': round(best_metrics['precision'], 4),
        'recall': round(best_metrics['recall'], 4),
        'f1_score': round(best_metrics['f1'], 4),
        'f2_score': round(best_metrics['f2'], 4),
        'auc_roc': round(auc, 4),
        
        # Confusion matrix
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        
        # Error rates
        'false_positive_rate': round(fpr, 4),
        'false_negative_rate': round(fnr, 4),
        'specificity': round(1-fpr, 4),
        'sensitivity': round(best_metrics['recall'], 4),
        
        # Business metrics
        'frauds_caught': int(tp),
        'frauds_missed': int(fn),
        'false_alarms': int(fp),
        
        # Model config
        'decision_threshold': round(best_threshold, 2),
        'scale_pos_weight': round(adaptive_weight, 1),
        'max_depth': int(params['max_depth']),
        'learning_rate': round(params['learning_rate'], 4),
        'n_estimators': int(params['n_estimators']),
        'training_time_seconds': round(training_time, 2)
    }
    
    # Write to log file (append)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(metrics_to_log) + '\n')
    
    print(f"Log written to: {LOG_FILE}")
    print(f"{'='*70}\n")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    train_fraud_model()



"""

for ($i=1; $i -le 20; $i++) { python fraud_training.py; Start-Sleep -Seconds 2 }

"""