### Lab: CI/CD for ML Infrastructure + AWS SAM CLI + AWS Lambda + GitHub Actions

### Quick Overview:
- Loads a pre-trained sklearn model (e.g., iris classification or simple regression)
- Accepts JSON input via API Gateway or direct invocation
- Returns prediction + confidence/metadata
- Demonstrates clean service-layer separation

1. Real ML deployment pattern (pickle → Lambda)
2. Shows how to handle model artifacts in serverless

---
### Core AWS SAM CLI concepts:
```
template.yaml (blueprint)
    ↓
sam build (packages code + dependencies)
    ↓
sam deploy (creates AWS resources)
    ↓
Lambda function running in AWS
```
- Resources section - Define Lambda function
- Properties - Configure function settings
- CodeUri - What gets packaged
- Handler - Entry point (lambda_function.lambda_handler)
- Environment - Pass config to Lambda

**What sam deploy Does**:
```
.aws-sam/build/
    ↓ Zips the package
    ↓ Uploads to S3
    ↓ Creates CloudFormation stack
    ↓
AWS Resources Created:
├── Lambda function
├── IAM execution role
├── CloudWatch log group
└── (Optional) API Gateway
```
1. SAM reads `template.yaml`
2. Finds `requirements.txt`
3. Creates `.aws-sam/build/` directory
4. Installs dependencies in isolated environment
5. Copies your code (src/, models/)
6. Packages everything together
7. `samconfig.toml` saves deployment settings (no need for --guided every time). SAM reads this for sam deploy parameters


### Actual SAM usage:
=========================
- Build: `sam build`
- What was packaged: `dir .aws-sam\build\MLInferenceFunction`
- `sam local invoke MLInferenceFunction --event event/test_event.json`
- `sam local invoke MLInferenceFunction --event event/api_gateway_event.json`
- Invoke Function: `sam local invoke`
- Test Function in the Cloud: `sam sync --stack-name {{stack-name}} --watch`
- If local deploy needed for learning: Deploy: `sam deploy --guided`


### What the CI/CD Pipeline Does:
GitHub Actions triggers:
  1. Checkout code
  2. Set up Python 3.11
  3. Install dependencies
  4. Run pytest (quality gate)
  5. Install AWS SAM CLI
  6. sam build
  7. sam deploy (to AWS Lambda)


---






### Code Structure:
┌─────────────────────────────────────┐
│   lambda_function.py (Handler)      │  ← AWS entry point, thin adapter
│   - Parse event                      │
│   - Call service                     │
│   - Return response                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   inference_service.py (Business)   │  ← Testable core logic
│   - Validate input                   │
│   - Get model from loader            │
│   - Run inference                    │
│   - Format response                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   model_loader.py (Singleton)       │  ← Lazy load, cache model
│   - Load model.pkl once              │
│   - Return cached instance           │
└───────────────────────────────────────┘

- template.yaml structure (Resources, Properties)
- sam build → packages dependencies
- sam deploy --guided → interactive setup
- Cleaner than raw aws lambda update-function-code
- template.yaml is Infrastructure-as-Code (auditable)

#### Pickle File Contents:
- A serialized Python object containing: model_object, metadata (e.g., version, training info)
- model_object parameters or weights. estimators, feature names, etc.
What Lambda does with it:
- Load pickle from disk → memory (cold start, ~100ms)
- Cache in memory for subsequent calls (warm start, ~0ms)
- Call model.predict(input) → instant results


###  Deployed Lambda Folder Structure:
```
/var/task/                    ← This is the root! Not named "CICD_ML_Infr_Lambda"
├── src/
├── models/
```