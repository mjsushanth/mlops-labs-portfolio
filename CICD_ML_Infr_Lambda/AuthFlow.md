
### GitHub Actions Deploys Lambda
```
Step 1: Workflow Starts
├── GitHub Actions runner starts
└── Reads secrets from GitHub Secrets

Step 2: AWS Authentication
├── Action: aws-actions/configure-aws-credentials@v4
├── Sets: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
└── These are now available to ALL subsequent steps

Step 3: SAM Build (Local)
├── sam build runs on GitHub runner
├── Packages code (no AWS contact yet)
└── Creates .aws-sam/build/ directory

Step 4: SAM Deploy - First AWS Contact
├── sam deploy command executes
├── AWS SDK reads credentials from environment variables
├── Makes API call: sts.GetCallerIdentity()
│   Request:
│   {
│     "AccessKeyId": "AKIAIOSFODNN7EXAMPLE",
│     "SecretAccessKey": "wJalrXUtn...",
│     "SignedHeaders": "..."
│   }
│
└── AWS Response:
    {
      "Account": "123456789012",           ← YOUR account!
      "UserId": "AIDAI23HXS...X7EXAMPLE",
      "Arn": "arn:aws:iam::123456789012:user/github-actions-deployer"
    }

Step 5: CloudFormation Stack Creation
├── SAM converts template.yaml to CloudFormation
├── Calls: cloudformation.CreateStack()
│   Request sent to:
│   - Account: 123456789012 (from credentials)
│   - Region: us-east-1 (from AWS_REGION)
│   - StackName: mini-ml-lambda-stack
│
└── CloudFormation creates resources IN YOUR ACCOUNT

Step 6: Lambda Function Creation
├── CloudFormation calls: lambda.CreateFunction()
├── Function created in:
│   - Account: 123456789012
│   - Region: us-east-1
│   - Name: mini-ml-lambda-stack-MLInferenceFunction-ABC123
│
└── Lambda is now in YOUR AWS account, accessible via YOUR console
```


### AWS Access Key Structure:

```
AKIAIOSFODNN7EXAMPLE
│││└─────────────────── Random identifier
││└─────────────────────── Key type
│└──────────────────────── AWS service prefix
└───────────────────────── Always starts with "A" for access key

Internally, AWS maps this to:
- Account ID: 123456789012
- IAM User: github-actions-deployer
- Permissions: Lambda, S3, CloudFormation, IAM
```