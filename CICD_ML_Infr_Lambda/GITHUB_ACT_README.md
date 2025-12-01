## GitHub Actions CI/CD Pipeline


#### Trigger Section:

```yaml
on:
  push:
    branches:
      - main
  workflow_dispatch:
```

#### Jobs Section:
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
```
```yaml
deploy:
    needs: test  # Only runs if test job passes
```

#### Key part:
```yaml
needs: test  # Dependency - test must pass first
```

