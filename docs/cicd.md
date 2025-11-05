---
title: "CI/CD & Automation"
---

# Chapter 7B: CI/CD & Automation

Manual deployments don't scale. Every code change should be automatically tested, security-scanned, and deployed to production. In this chapter, I'll show you how to build a complete CI/CD pipeline for ResearcherAI using GitHub Actions, with automated testing, security scanning, and GitOps-based deployments.

## Why CI/CD for ML/AI Applications?

Traditional web apps need CI/CD, but LLM applications have additional challenges:

**Traditional CI/CD Concerns:**
- Code quality (linting, formatting)
- Unit tests
- Integration tests
- Security vulnerabilities
- Build artifacts
- Deployment automation

**LLM-Specific Concerns:**
- Prompt regression testing
- Model version management
- Token usage regression
- Cost regression testing
- Response quality evaluation
- Embedding drift detection
- Vector database migrations

### The Goal

```
git push
  ↓
Automated Pipeline
  ├─ Lint & format code
  ├─ Run unit tests (Python)
  ├─ Run LLM evaluation tests
  ├─ Check prompt regressions
  ├─ Build Docker images
  ├─ Scan for vulnerabilities
  ├─ Deploy to staging
  ├─ Run integration tests
  ├─ Deploy to production
  └─ Monitor deployment
```

## GitHub Actions Overview

GitHub Actions is like having a CI/CD server that runs automatically on every git push.

### Web Developer Analogy

**GitHub Actions** = Automated deployment like Vercel/Netlify
- Triggers on git push
- Runs tests automatically
- Builds and deploys
- All configured in `.github/workflows/`

**Workflow** = Recipe for automation
**Job** = Group of related steps
**Step** = Individual command or action
**Action** = Reusable workflow component

### Basic Structure

`.github/workflows/ci.yml`:

```yaml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest
```

## Complete CI/CD Pipeline for ResearcherAI

Let's build a production-grade pipeline step by step.

### Step 1: Code Quality Checks

`.github/workflows/lint.yml`:

```yaml
name: Code Quality

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install black flake8 mypy pylint isort
          pip install -r requirements.txt

      - name: Check code formatting (Black)
        run: black --check .

      - name: Check import sorting (isort)
        run: isort --check-only .

      - name: Lint with flake8
        run: |
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

      - name: Type checking (mypy)
        run: mypy agents/ utils/ --ignore-missing-imports

      - name: Lint with pylint
        run: pylint agents/ utils/ --disable=C0111,C0103

  yaml-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Lint YAML files
        uses: ibiqlik/action-yamllint@v3
        with:
          file_or_dir: .github/workflows/*.yml k8s/ terraform/

  dockerfile-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Lint Dockerfile
        uses: hadolint/hadolint-action@v3.1.0
        with:
          dockerfile: Dockerfile
```

### Step 2: Automated Testing

`.github/workflows/test.yml`:

```yaml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio pytest-mock

      - name: Run unit tests
        run: |
          pytest tests/unit/ \
            --cov=agents \
            --cov=utils \
            --cov-report=xml \
            --cov-report=html \
            --junitxml=test-results.xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: test-results-${{ matrix.python-version }}
          path: test-results.xml

  integration-tests:
    runs-on: ubuntu-latest
    services:
      neo4j:
        image: neo4j:5.13-community
        env:
          NEO4J_AUTH: neo4j/testpassword
        ports:
          - 7687:7687
          - 7474:7474

      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333

      kafka:
        image: bitnami/kafka:latest
        env:
          KAFKA_ENABLE_KRAFT: yes
          KAFKA_CFG_PROCESS_ROLES: broker,controller
          KAFKA_CFG_CONTROLLER_LISTENER_NAMES: CONTROLLER
          KAFKA_CFG_LISTENERS: PLAINTEXT://:9092,CONTROLLER://:9093
          KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
          KAFKA_CFG_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
          KAFKA_BROKER_ID: 1
          KAFKA_CFG_CONTROLLER_QUORUM_VOTERS: 1@localhost:9093
          ALLOW_PLAINTEXT_LISTENER: yes
        ports:
          - 9092:9092

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio

      - name: Wait for services
        run: |
          timeout 60 bash -c 'until nc -z localhost 7687; do sleep 1; done'
          timeout 60 bash -c 'until nc -z localhost 6333; do sleep 1; done'
          timeout 60 bash -c 'until nc -z localhost 9092; do sleep 1; done'

      - name: Run integration tests
        env:
          NEO4J_URI: bolt://localhost:7687
          NEO4J_PASSWORD: testpassword
          QDRANT_HOST: localhost:6333
          KAFKA_BOOTSTRAP_SERVERS: localhost:9092
        run: |
          pytest tests/integration/ -v

  llm-evaluation-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install langsmith

      - name: Run LLM evaluation tests
        env:
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
          LANGCHAIN_PROJECT: ResearcherAI-CI
        run: |
          pytest tests/llm_evaluation/ -v --tb=short

      - name: Check for prompt regressions
        run: |
          python scripts/check_prompt_regressions.py \
            --baseline=prompts/baseline.json \
            --current=prompts/current.json

      - name: Check token usage regression
        run: |
          python scripts/check_token_regression.py \
            --max-increase-percent=10
```

### Step 3: Security Scanning

`.github/workflows/security.yml`:

```yaml
name: Security Scanning

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Snyk to check for vulnerabilities
        uses: snyk/actions/python@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          command: test
          args: --severity-threshold=high

      - name: Safety check (Python dependencies)
        run: |
          pip install safety
          safety check --json

  secret-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for secret scanning

      - name: TruffleHog Secret Scanning
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: main
          head: HEAD

      - name: GitLeaks Secret Scanning
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  code-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: python

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2

  container-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t researcherai:${{ github.sha }} .

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: researcherai:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

### Step 4: Build and Push Docker Images

`.github/workflows/build.yml`:

```yaml
name: Build and Push

on:
  push:
    branches: [main, develop]
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64

      - name: Image digest
        run: echo ${{ steps.meta.outputs.digest }}
```

### Step 5: Deploy to Staging

`.github/workflows/deploy-staging.yml`:

```yaml
name: Deploy to Staging

on:
  push:
    branches: [develop]
  workflow_dispatch:

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    environment:
      name: staging
      url: https://staging.researcherai.com

    steps:
      - uses: actions/checkout@v4

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG_STAGING }}

      - name: Install Helm
        uses: azure/setup-helm@v3
        with:
          version: '3.12.0'

      - name: Deploy with Helm
        run: |
          helm upgrade --install researcherai ./k8s/helm/researcherai \
            --namespace staging \
            --create-namespace \
            --set app.image.tag=${{ github.sha }} \
            --set app.secrets.googleApiKey=${{ secrets.GOOGLE_API_KEY_STAGING }} \
            --set app.environment=staging \
            --wait \
            --timeout 10m

      - name: Verify deployment
        run: |
          kubectl rollout status deployment/researcherai -n staging
          kubectl get pods -n staging

      - name: Run smoke tests
        run: |
          sleep 30  # Wait for services to be ready
          curl -f https://staging.researcherai.com/health || exit 1
          python scripts/smoke_tests.py --environment=staging

      - name: Notify Slack
        if: always()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'Staging deployment ${{ job.status }}'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Step 6: Deploy to Production

`.github/workflows/deploy-production.yml`:

```yaml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:

jobs:
  deploy-production:
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://researcherai.com

    steps:
      - uses: actions/checkout@v4

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG_PRODUCTION }}

      - name: Install Helm
        uses: azure/setup-helm@v3

      - name: Blue-Green Deployment
        run: |
          # Deploy to blue environment
          helm upgrade --install researcherai-blue ./k8s/helm/researcherai \
            --namespace production \
            --create-namespace \
            --set app.image.tag=${{ github.ref_name }} \
            --set app.secrets.googleApiKey=${{ secrets.GOOGLE_API_KEY_PROD }} \
            --set app.environment=production \
            --set app.deployment.color=blue \
            --wait \
            --timeout 10m

      - name: Run production smoke tests
        run: |
          python scripts/smoke_tests.py \
            --environment=production \
            --target=blue \
            --critical-only

      - name: Switch traffic to blue
        run: |
          kubectl patch service researcherai -n production \
            -p '{"spec":{"selector":{"color":"blue"}}}'

      - name: Monitor for 5 minutes
        run: |
          sleep 300
          python scripts/monitor_deployment.py \
            --duration=300 \
            --error-threshold=0.01

      - name: Scale down green
        if: success()
        run: |
          helm delete researcherai-green -n production || true

      - name: Rollback on failure
        if: failure()
        run: |
          kubectl patch service researcherai -n production \
            -p '{"spec":{"selector":{"color":"green"}}}'
          helm delete researcherai-blue -n production

      - name: Create GitHub Release
        if: success()
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref_name }}
          body: |
            ## Changes
            ${{ github.event.head_commit.message }}

            ## Deployment
            - Deployed to production
            - Image: ghcr.io/${{ github.repository }}:${{ github.ref_name }}

      - name: Notify Slack
        if: always()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'Production deployment ${{ job.status }}: ${{ github.ref_name }}'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## GitOps with ArgoCD

For declarative deployments, ArgoCD watches your Git repository and automatically syncs changes to Kubernetes.

### Step 1: Install ArgoCD

```bash
# Create namespace
kubectl create namespace argocd

# Install ArgoCD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Get admin password
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}" | base64 -d

# Port forward to access UI
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Login
argocd login localhost:8080
```

### Step 2: Create ArgoCD Application

`argocd/application.yaml`:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: researcherai
  namespace: argocd
spec:
  project: default

  source:
    repoURL: https://github.com/your-org/ResearcherAI.git
    targetRevision: HEAD
    path: k8s/helm/researcherai
    helm:
      valueFiles:
        - values.yaml
        - values-production.yaml

  destination:
    server: https://kubernetes.default.svc
    namespace: production

  syncPolicy:
    automated:
      prune: true      # Delete resources not in Git
      selfHeal: true   # Sync on changes
    syncOptions:
      - CreateNamespace=true

    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

Apply:

```bash
kubectl apply -f argocd/application.yaml -n argocd
```

### Step 3: Update GitHub Actions for GitOps

`.github/workflows/gitops-deploy.yml`:

```yaml
name: GitOps Deployment

on:
  push:
    branches: [main]
    paths:
      - 'k8s/**'
      - 'Dockerfile'

jobs:
  update-manifests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Update image tag
        run: |
          sed -i "s|tag: .*|tag: ${GITHUB_SHA}|" \
            k8s/helm/researcherai/values-production.yaml

      - name: Commit changes
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add k8s/helm/researcherai/values-production.yaml
          git commit -m "Update production image to ${GITHUB_SHA}"
          git push

# ArgoCD automatically detects the change and syncs!
```

## LLM-Specific CI/CD Practices

### 1. Prompt Regression Testing

`scripts/check_prompt_regressions.py`:

```python
#!/usr/bin/env python3
import json
import sys
from langsmith import Client

def check_prompt_regressions():
    """
    Compare current prompts against baseline.
    Fail if quality drops significantly.
    """
    client = Client()

    # Load baseline scores
    with open("prompts/baseline.json") as f:
        baseline = json.load(f)

    # Run current evaluation
    results = client.evaluate(
        lambda x: generate_with_current_prompt(x),
        data="research-queries-test-set",
        evaluators=[quality_evaluator, relevance_evaluator]
    )

    # Compare scores
    for metric in ["quality", "relevance"]:
        current_score = results[metric]
        baseline_score = baseline[metric]

        # Fail if more than 5% drop
        if current_score < baseline_score * 0.95:
            print(f"❌ Regression detected in {metric}!")
            print(f"  Baseline: {baseline_score:.2f}")
            print(f"  Current:  {current_score:.2f}")
            print(f"  Drop:     {(baseline_score - current_score):.2f}")
            sys.exit(1)

    print("✅ No prompt regressions detected")

if __name__ == "__main__":
    check_prompt_regressions()
```

### 2. Token Usage Monitoring

`scripts/check_token_regression.py`:

```python
#!/usr/bin/env python3
import sys
from langfuse import Langfuse

def check_token_regression(max_increase_percent=10):
    """
    Ensure token usage doesn't increase unexpectedly.
    """
    langfuse = Langfuse()

    # Get token usage from last week
    last_week = langfuse.get_trace_stats(
        filter_time_start="7d ago",
        filter_time_end="now"
    )

    avg_tokens_last_week = last_week["avg_tokens_per_trace"]

    # Get token usage from last day
    last_day = langfuse.get_trace_stats(
        filter_time_start="1d ago",
        filter_time_end="now"
    )

    avg_tokens_last_day = last_day["avg_tokens_per_trace"]

    # Calculate increase
    increase_percent = (
        (avg_tokens_last_day - avg_tokens_last_week) / avg_tokens_last_week
    ) * 100

    if increase_percent > max_increase_percent:
        print(f"❌ Token usage increased by {increase_percent:.1f}%!")
        print(f"  Last week: {avg_tokens_last_week:.0f} tokens")
        print(f"  Last day:  {avg_tokens_last_day:.0f} tokens")
        sys.exit(1)

    print(f"✅ Token usage stable (change: {increase_percent:.1f}%)")

if __name__ == "__main__":
    check_token_regression()
```

### 3. Cost Budget Checks

`scripts/check_cost_budget.py`:

```python
#!/usr/bin/env python3
import sys
from langfuse import Langfuse

def check_cost_budget(daily_budget=100):
    """
    Fail deployment if projected cost exceeds budget.
    """
    langfuse = Langfuse()

    # Get cost from last hour
    stats = langfuse.get_trace_stats(
        filter_time_start="1h ago",
        filter_time_end="now"
    )

    hourly_cost = stats["total_cost"]
    projected_daily_cost = hourly_cost * 24

    if projected_daily_cost > daily_budget:
        print(f"❌ Projected daily cost exceeds budget!")
        print(f"  Budget:    ${daily_budget:.2f}")
        print(f"  Projected: ${projected_daily_cost:.2f}")
        print(f"  Hourly:    ${hourly_cost:.2f}")
        sys.exit(1)

    print(f"✅ Cost within budget (${projected_daily_cost:.2f}/day)")

if __name__ == "__main__":
    check_cost_budget()
```

## Deployment Strategies

### 1. Rolling Update (Default)

```yaml
# k8s/helm/researcherai/templates/deployment.yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Add 1 pod before removing old
      maxUnavailable: 0  # Always keep all pods running
```

**Pros:**
- Zero downtime
- Automatic rollback on failure
- Resource efficient

**Cons:**
- Gradual rollout (not instant)
- Mixed versions running temporarily

### 2. Blue-Green Deployment

```yaml
# Deploy blue
helm install researcherai-blue ./k8s/helm/researcherai \
  --set app.deployment.color=blue

# Test blue
curl https://blue.researcherai.com/health

# Switch traffic
kubectl patch service researcherai \
  -p '{"spec":{"selector":{"color":"blue"}}}'

# Remove green
helm delete researcherai-green
```

**Pros:**
- Instant rollback
- Full testing before switch
- Clean separation

**Cons:**
- Requires 2x resources
- Database migrations complex

### 3. Canary Deployment

```yaml
# Deploy canary with 10% traffic
helm install researcherai-canary ./k8s/helm/researcherai \
  --set app.replicaCount=1 \
  --set ingress.weight=10

# Monitor
python scripts/monitor_canary.py --duration=600

# If successful, scale up
helm upgrade researcherai-canary \
  --set app.replicaCount=10 \
  --set ingress.weight=100
```

**Pros:**
- Gradual rollout
- Early problem detection
- User-based routing possible

**Cons:**
- Complex traffic management
- Requires service mesh (Istio)

## Monitoring CI/CD Pipeline

### GitHub Actions Dashboard

View all workflow runs:
```
https://github.com/your-org/ResearcherAI/actions
```

### Pipeline Metrics to Track

1. **Build Time**
   - Target: < 10 minutes
   - Alert if > 15 minutes

2. **Test Pass Rate**
   - Target: 100%
   - Alert if < 95%

3. **Deployment Frequency**
   - Target: Multiple times per day
   - Track trend

4. **Mean Time to Recovery (MTTR)**
   - Target: < 1 hour
   - Track incidents

5. **Change Failure Rate**
   - Target: < 5%
   - Track rollbacks

### Pipeline Alerts

`.github/workflows/alerts.yml`:

```yaml
name: Pipeline Alerts

on:
  workflow_run:
    workflows: ["CI Pipeline", "Deploy to Production"]
    types: [completed]

jobs:
  alert-on-failure:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'failure' }}
    steps:
      - name: Send PagerDuty alert
        uses: PagerDuty/pagerduty-github-action@v1
        with:
          pagerduty-token: ${{ secrets.PAGERDUTY_TOKEN }}
          service-id: ${{ secrets.PAGERDUTY_SERVICE_ID }}
          event-action: trigger
          summary: "CI/CD Pipeline failed: ${{ github.event.workflow_run.name }}"
          severity: high
```

## Best Practices

### 1. Fast Feedback

```yaml
# Run fast tests first
jobs:
  quick-tests:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/unit/ -v  # < 2 minutes

  slow-tests:
    needs: quick-tests  # Only run if quick tests pass
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/integration/ -v  # 5-10 minutes
```

### 2. Fail Fast

```yaml
jobs:
  test:
    strategy:
      fail-fast: true  # Stop all jobs on first failure
      matrix:
        python-version: ['3.10', '3.11', '3.12']
```

### 3. Caching

```yaml
- name: Cache pip packages
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-

- name: Cache Docker layers
  uses: actions/cache@v3
  with:
    path: /tmp/.buildx-cache
    key: ${{ runner.os }}-buildx-${{ github.sha }}
    restore-keys: |
      ${{ runner.os }}-buildx-
```

### 4. Secure Secrets

```yaml
# Never log secrets
- name: Deploy with secrets
  run: |
    echo "::add-mask::${{ secrets.API_KEY }}"
    deploy.sh --api-key="${{ secrets.API_KEY }}"
  # API_KEY is now masked in logs
```

### 5. Environment Protection

```yaml
# Require approval for production
environment:
  name: production
  url: https://researcherai.com
# Configure in GitHub Settings → Environments
# - Require reviewers: 2 approvals
# - Wait timer: 0 minutes
# - Deployment branches: main only
```

## Troubleshooting CI/CD

### Pipeline Fails on Dependencies

```yaml
# Use exact versions
- name: Install dependencies
  run: |
    pip install -r requirements-lock.txt  # Pinned versions
    # Not: pip install -r requirements.txt  # Can break
```

### Tests Pass Locally, Fail in CI

```yaml
# Use same environment
- name: Run tests in Docker
  run: |
    docker-compose -f docker-compose.test.yml up --abort-on-container-exit
    # Same environment as local development
```

### Slow Pipelines

```yaml
# Run jobs in parallel
jobs:
  unit-tests:
    runs-on: ubuntu-latest
  integration-tests:
    runs-on: ubuntu-latest
  security-scan:
    runs-on: ubuntu-latest
# All run simultaneously
```

### Deployment Failures

```yaml
# Add rollback step
- name: Rollback on failure
  if: failure()
  run: |
    helm rollback researcherai -n production
    kubectl rollout undo deployment/researcherai -n production
```

## Complete CI/CD Checklist

### ✅ Code Quality
- [ ] Linting (black, flake8)
- [ ] Type checking (mypy)
- [ ] Import sorting (isort)
- [ ] Complexity checks (pylint)

### ✅ Testing
- [ ] Unit tests (>80% coverage)
- [ ] Integration tests
- [ ] LLM evaluation tests
- [ ] Prompt regression tests
- [ ] Token usage checks
- [ ] Cost budget checks

### ✅ Security
- [ ] Dependency scanning (Snyk)
- [ ] Secret scanning (TruffleHog)
- [ ] Code scanning (CodeQL)
- [ ] Container scanning (Trivy)
- [ ] License compliance

### ✅ Build & Deploy
- [ ] Docker image building
- [ ] Multi-arch builds (amd64, arm64)
- [ ] Image signing
- [ ] Helm chart validation
- [ ] Kubernetes deployment
- [ ] Smoke tests
- [ ] Health checks

### ✅ Monitoring
- [ ] Deployment tracking
- [ ] Rollback automation
- [ ] Slack/PagerDuty alerts
- [ ] Metrics collection
- [ ] Log aggregation

### ✅ GitOps (Optional)
- [ ] ArgoCD setup
- [ ] Declarative manifests
- [ ] Automated sync
- [ ] Self-healing

## Conclusion

Congratulations! You now have a complete CI/CD pipeline that:

- ✅ Automatically tests every code change
- ✅ Checks for security vulnerabilities
- ✅ Validates prompt quality
- ✅ Monitors token usage and costs
- ✅ Builds and pushes Docker images
- ✅ Deploys to staging and production
- ✅ Runs smoke tests
- ✅ Monitors deployments
- ✅ Automatically rolls back on failure
- ✅ Sends alerts on issues

Your ResearcherAI system is now production-ready with:
1. **Kubernetes deployment** - Auto-scaling, high availability
2. **Terraform infrastructure** - Reproducible cloud resources
3. **Comprehensive observability** - LangFuse, Prometheus, Grafana
4. **Automated CI/CD** - GitHub Actions, security scanning
5. **Multiple environments** - Dev, staging, production

In the final chapter, we'll wrap up with lessons learned and future enhancements!
