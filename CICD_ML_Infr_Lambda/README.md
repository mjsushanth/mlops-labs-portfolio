### Lab: CI/CD for ML Infrastructure + AWS SAM CLI + AWS Lambda + GitHub Actions

### Quick Overview:
- Load a sklearn model or dataset (e.g., iris classification)
- Setup Training, Saved model. Setup models/ , src/ and tests/, practice with PyTests and see coverage.
- After ML setup - ensure **proper serving and inference** files, use singleton model instance and caching, lazy loading, optimizations.
- Understand and explore with **SAM Infrastructure**, **SAM+Docker dependancy**, **SAM invoke, SAM build** and tests.
- Understand **AWS IAM** users and policies, **Github PAT** tokens, **AWS Secrets setup** and configure carefully.
- Perfect workflows on Github ACTIONS, **understand Github Actions -> AWS communication.**
- Setup AWS LAMBDA: Accepts JSON input via API Gateway or direct invocation
- Returns prediction + confidence/metadata
- Demonstrate clean service-layer separation

1. Real ML deployment pattern (pickle → Lambda)
2. Shows how to handle model artifacts in **serverless**


### What this quick Mini-Project / Lab covers:
```
Complete:
├── ML Model Training (train_model.py)
├── Clean Architecture (3-layer separation)
├── Path Resolution (root detection pattern)
├── Singleton Pattern (cold start optimization)
├── Comprehensive Tests (pytest suite)
├── SAM Infrastructure (template.yaml)
├── GitHub Actions (deploy-ml-lambda.yml)
├── AWS Lambda (production deployment)
└── Documentation (README.md)
```

### Please check results at:
- [View Screenshots](results_screenshots)

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


### IMPORTANT: Assume you've written all code, src, models, 'serving' files, tests, and also tested SAM CLI locally. There will be still steps to be done on the GitHub page for a token and also steps to be done on the AWS console to generate a respective user and secret IDs and keys: 
- For the last phase on GitHub Actions, you need to set up AWS credentials as GitHub Secrets:
  - AWS_ACCESS_KEY_ID
  - AWS_SECRET_ACCESS_KEY
  - AWS_REGION
- To get these, you have to go to your respective AWS console, access the IAM service, and make sure that you first create a user. When you do create a user, select four important policies here.
  - AWSLambda_FullAccess
  - IAMFullAccess
  - AmazonS3FullAccess
  - AWSCloudFormationFullAccess
- After creating the user, click on that particular user created option and you can easily access the security credentials or the quick button to create access key ID and secret access key. Please copy them carefully.

### Generating GitHub Personal Access Token (PAT):
- In GitHub, you need to go and access the settings and in your settings, 
  - Personal Access Tokens page: `https://github.com/settings/tokens`
  - Create new classic token directly: `https://github.com/settings/tokens/new`
1. Sign in to GitHub.com.
2. Click your profile photo (top-right) → Settings.
3. In the left sidebar, open Developer settings.
4. Click Personal access tokens → Tokens (classic).
5. From here you can: generate a new classic token (Generate new token (classic)), view the list of tokens, revoke/delete, or regenerate.
6. To create a token, click Generate new token (classic), choose expiry and scopes, then create. Note: You will only see the token value once—copy and store it safely.
7. For this particular item, make sure that you select minimal scopes or feel free to select the workflow and rest of the important scopes. Admin can also be selected, but it is not necessary. We have tested. The main scopes are repo, workflows, write packages, etc.
8. Once again, after you are done with this process, you have to add it to your Windows credentials. 
   1. GitHub credential: `git:https://github.com`

--- 



### Code Structure:
```
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
```

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
