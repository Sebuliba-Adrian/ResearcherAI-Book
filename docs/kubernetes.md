---
title: "Kubernetes & Helm Deployment"
---

# Chapter 6A: Kubernetes & Helm Deployment

After building and testing our multi-agent RAG system, we need a production-grade deployment strategy. In this chapter, I'll walk you through deploying ResearcherAI to Kubernetes using Helm charts - taking it from development Docker Compose to cloud-native, auto-scaling infrastructure.

## Why Kubernetes for ResearcherAI?

Before diving into implementation, let me explain why I chose Kubernetes for production deployment:

### The Challenge

Our ResearcherAI system has complex requirements:
- **6 different services**: Application, Neo4j, Qdrant, Kafka (3 brokers), Zookeeper (3 nodes)
- **Stateful data**: Graph database and vector database need persistent storage
- **Event streaming**: Kafka cluster requires coordination and high availability
- **Variable load**: Research queries can spike unpredictably
- **Resource management**: LLM API calls are expensive, need cost controls

### Docker Compose vs. Kubernetes

**Docker Compose (Development)**
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - neo4j
      - qdrant
      - kafka
```

This works great for development, but in production we need:
- Auto-scaling when traffic increases
- Automatic restarts when services crash
- Rolling updates with zero downtime
- Resource limits to control costs
- Health checks and monitoring
- Multiple replicas for high availability

### Web Developer Analogy

Think of Kubernetes like moving from a single server to a managed platform:

**Docker Compose** = Deploying your Node.js app on a single VPS
- You SSH in, run `docker-compose up`
- Works great until traffic increases
- Manual scaling, manual recovery

**Kubernetes** = Deploying to a platform like Heroku or Vercel
- Platform handles scaling automatically
- Self-healing when things break
- Built-in load balancing
- But more complex to set up

## Kubernetes Basics for Web Developers

If you're coming from web development, here's how Kubernetes concepts map to what you know:

### Core Concepts

**1. Pods** (Like containers, but smarter)
```javascript
// In Express, you run one server:
const app = express();
app.listen(3000);

// In Kubernetes, a Pod runs one or more containers together:
Pod {
  containers: [app, sidecar]  // Usually just one
  shared_network: true
  shared_storage: true
}
```

**2. Deployments** (Like PM2 or Forever)
```javascript
// PM2 keeps your app running:
pm2 start app.js -i 4  // 4 instances

// Kubernetes Deployment does the same:
Deployment {
  replicas: 4
  auto_restart: true
  rolling_updates: true
}
```

**3. Services** (Like Nginx reverse proxy)
```nginx
# Nginx load balances to backends:
upstream backend {
    server 10.0.0.1:8000;
    server 10.0.0.2:8000;
}

# Kubernetes Service does this automatically:
Service {
  type: LoadBalancer
  selects: Pods with label "app=researcherai"
  distributes: traffic to all selected Pods
}
```

**4. ConfigMaps** (Like .env files)
```bash
# .env file
DATABASE_URL=postgres://localhost/db
API_KEY=secret123

# Kubernetes ConfigMap
ConfigMap {
  data:
    DATABASE_URL: postgres://localhost/db
    # Secrets go in separate Secret resource
}
```

**5. Persistent Volumes** (Like mounted volumes)
```yaml
# Docker Compose volume:
volumes:
  - ./data:/app/data

# Kubernetes PersistentVolumeClaim:
PersistentVolumeClaim {
  storage: 10Gi
  accessMode: ReadWriteOnce
}
```

## Helm: Kubernetes Package Manager

Helm is to Kubernetes what npm is to Node.js:

**npm** (Node.js packages):
```bash
npm install express
npm install -g pm2
```

**Helm** (Kubernetes packages):
```bash
helm install my-app ./chart
helm upgrade my-app ./chart
```

### Helm Chart Structure

```
researcherai/              # Like package.json + all code
├── Chart.yaml             # Package metadata (like package.json)
├── values.yaml            # Configuration (like .env.example)
├── templates/             # Kubernetes YAML files (like src/)
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
└── charts/                # Dependencies (like node_modules/)
```

## Building the ResearcherAI Helm Chart

Now let's build our production deployment step by step.

### Step 1: Chart Metadata

First, I created `Chart.yaml` to define the package:

```yaml
apiVersion: v2
name: researcherai
description: Production-grade Multi-Agent RAG System
type: application
version: 2.0.0        # Chart version
appVersion: "2.0.0"   # Application version
keywords:
  - rag
  - multi-agent
  - knowledge-graph
  - vector-search

# Dependencies (like npm dependencies)
dependencies:
  - name: neo4j
    version: 5.13.0
    repository: https://helm.neo4j.com/neo4j
    condition: neo4j.enabled

  - name: qdrant
    version: 0.7.0
    repository: https://qdrant.github.io/qdrant-helm
    condition: qdrant.enabled

  - name: strimzi-kafka-operator
    version: 0.38.0
    repository: https://strimzi.io/charts/
    condition: kafka.enabled
```

**Why dependencies?**
- Neo4j and Qdrant have official Helm charts
- No need to reinvent the wheel
- Automatic updates and best practices
- But we still control configuration through our `values.yaml`

### Step 2: Configuration Values

`values.yaml` is like `.env.example` - all configurable settings:

```yaml
# Global settings
global:
  namespace: researcherai
  storageClass: standard  # Cloud provider's default storage
  imagePullSecrets: []

# Application configuration
app:
  name: researcherai
  replicaCount: 2  # Start with 2 replicas for high availability

  image:
    repository: researcherai/multiagent
    tag: "2.0.0"
    pullPolicy: IfNotPresent

  # Resource limits (prevent runaway costs!)
  resources:
    requests:
      cpu: 1000m      # 1 CPU core
      memory: 2Gi     # 2GB RAM
    limits:
      cpu: 2000m      # Max 2 CPU cores
      memory: 4Gi     # Max 4GB RAM

  # Auto-scaling configuration
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 80
    targetMemoryUtilizationPercentage: 80

  # Health checks
  health:
    livenessProbe:
      path: /health
      initialDelaySeconds: 30
      periodSeconds: 30
    readinessProbe:
      path: /health
      initialDelaySeconds: 10
      periodSeconds: 10

# Neo4j configuration
neo4j:
  enabled: true
  image:
    tag: "5.13-community"

  # Neo4j memory tuning
  config:
    dbms.memory.heap.initial_size: "2G"
    dbms.memory.heap.max_size: "2G"
    dbms.memory.pagecache.size: "1G"

  # Enable APOC procedures
  apoc:
    core:
      enabled: true

  # Persistent storage
  persistentVolume:
    size: 10Gi
    storageClass: standard

# Qdrant configuration
qdrant:
  enabled: true
  replicaCount: 1

  resources:
    requests:
      cpu: 1000m
      memory: 2Gi
    limits:
      cpu: 2000m
      memory: 4Gi

  persistence:
    size: 10Gi
    storageClass: standard

# Kafka configuration
kafka:
  enabled: true
  cluster:
    name: rag-kafka
    version: 3.6.0
    replicas: 3  # 3 brokers for high availability

    # Storage per broker
    storage:
      type: persistent-claim
      size: 10Gi
      storageClass: standard

    # Kafka configuration
    config:
      offsets.topic.replication.factor: 3
      transaction.state.log.replication.factor: 3
      transaction.state.log.min.isr: 2
      default.replication.factor: 3
      min.insync.replicas: 2

  # Zookeeper configuration
  zookeeper:
    replicas: 3
    storage:
      type: persistent-claim
      size: 5Gi
      storageClass: standard
```

### Step 3: Application Deployment

The core deployment template (`templates/app-deployment.yaml`):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "researcherai.fullname" . }}
  namespace: {{ .Values.global.namespace }}
  labels:
    {{- include "researcherai.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.app.replicaCount }}

  # Rolling update strategy
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Add 1 extra pod during update
      maxUnavailable: 0  # Keep all pods running during update

  selector:
    matchLabels:
      {{- include "researcherai.selectorLabels" . | nindent 6 }}

  template:
    metadata:
      labels:
        {{- include "researcherai.selectorLabels" . | nindent 8 }}
      annotations:
        # Force restart when config changes
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
        checksum/secret: {{ include (print $.Template.BasePath "/secret.yaml") . | sha256sum }}

    spec:
      # Security context
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000

      # Init containers - wait for dependencies
      initContainers:
        # Wait for Neo4j
        - name: wait-for-neo4j
          image: busybox:1.35
          command:
            - sh
            - -c
            - |
              until nc -z {{ include "researcherai.neo4jHost" . }} 7687; do
                echo "Waiting for Neo4j..."
                sleep 2
              done

        # Wait for Qdrant
        - name: wait-for-qdrant
          image: busybox:1.35
          command:
            - sh
            - -c
            - |
              until nc -z {{ include "researcherai.qdrantHost" . }} 6333; do
                echo "Waiting for Qdrant..."
                sleep 2
              done

        # Wait for Kafka
        - name: wait-for-kafka
          image: busybox:1.35
          command:
            - sh
            - -c
            - |
              until nc -z {{ include "researcherai.kafkaBootstrap" . }} 9092; do
                echo "Waiting for Kafka..."
                sleep 2
              done

      # Main application container
      containers:
        - name: researcherai
          image: "{{ .Values.app.image.repository }}:{{ .Values.app.image.tag }}"
          imagePullPolicy: {{ .Values.app.image.pullPolicy }}

          # Container ports
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP

          # Environment variables from ConfigMap
          envFrom:
            - configMapRef:
                name: {{ include "researcherai.fullname" . }}-config
            - secretRef:
                name: {{ include "researcherai.fullname" . }}-secrets

          # Resource limits
          resources:
            {{- toYaml .Values.app.resources | nindent 12 }}

          # Liveness probe - restart if unhealthy
          livenessProbe:
            httpGet:
              path: {{ .Values.app.health.livenessProbe.path }}
              port: http
            initialDelaySeconds: {{ .Values.app.health.livenessProbe.initialDelaySeconds }}
            periodSeconds: {{ .Values.app.health.livenessProbe.periodSeconds }}
            timeoutSeconds: 5
            failureThreshold: 3

          # Readiness probe - don't send traffic if not ready
          readinessProbe:
            httpGet:
              path: {{ .Values.app.health.readinessProbe.path }}
              port: http
            initialDelaySeconds: {{ .Values.app.health.readinessProbe.initialDelaySeconds }}
            periodSeconds: {{ .Values.app.health.readinessProbe.periodSeconds }}
            timeoutSeconds: 3
            failureThreshold: 3

          # Volumes
          volumeMounts:
            - name: logs
              mountPath: /app/logs
            - name: sessions
              mountPath: /app/sessions

      # Volume definitions
      volumes:
        - name: logs
          emptyDir: {}
        - name: sessions
          emptyDir: {}
```

**Key Features:**

1. **Init Containers**: Wait for dependencies before starting
2. **Rolling Updates**: Zero-downtime deployments
3. **Health Checks**: Automatic restart if unhealthy
4. **Security Context**: Run as non-root user
5. **Resource Limits**: Prevent runaway costs
6. **Configuration Management**: Auto-restart on config changes

### Step 4: Auto-Scaling

Horizontal Pod Autoscaler (`templates/hpa.yaml`):

```yaml
{{- if .Values.app.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "researcherai.fullname" . }}
  namespace: {{ .Values.global.namespace }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "researcherai.fullname" . }}

  minReplicas: {{ .Values.app.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.app.autoscaling.maxReplicas }}

  metrics:
    # Scale based on CPU usage
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.app.autoscaling.targetCPUUtilizationPercentage }}

    # Scale based on memory usage
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: {{ .Values.app.autoscaling.targetMemoryUtilizationPercentage }}

  # Scaling behavior
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 minutes before scaling down
      policies:
        - type: Percent
          value: 50    # Remove max 50% of pods at once
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0    # Scale up immediately
      policies:
        - type: Percent
          value: 100   # Can double pods quickly
          periodSeconds: 15
{{- end }}
```

**How it works:**

```
Traffic Spike:
  CPU > 80% → Add pods (up to max 10)
  CPU < 80% for 5min → Remove pods (down to min 2)

Memory Pressure:
  Memory > 80% → Add pods
  Memory < 80% → Remove pods
```

### Step 5: Kafka Event Streaming

Kafka cluster configuration (`templates/kafka-cluster.yaml`):

```yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata:
  name: {{ .Values.kafka.cluster.name }}
  namespace: {{ .Values.global.namespace }}
spec:
  # Kafka brokers
  kafka:
    version: {{ .Values.kafka.cluster.version }}
    replicas: {{ .Values.kafka.cluster.replicas }}

    listeners:
      - name: plain
        port: 9092
        type: internal
        tls: false
      - name: tls
        port: 9093
        type: internal
        tls: true

    config:
      offsets.topic.replication.factor: {{ .Values.kafka.cluster.config.offsets.topic.replication.factor }}
      transaction.state.log.replication.factor: {{ .Values.kafka.cluster.config.transaction.state.log.replication.factor }}
      transaction.state.log.min.isr: {{ .Values.kafka.cluster.config.transaction.state.log.min.isr }}
      default.replication.factor: {{ .Values.kafka.cluster.config.default.replication.factor }}
      min.insync.replicas: {{ .Values.kafka.cluster.config.min.insync.replicas }}

      # Performance tuning
      num.network.threads: 8
      num.io.threads: 8
      socket.send.buffer.bytes: 102400
      socket.receive.buffer.bytes: 102400
      socket.request.max.bytes: 104857600

    # Storage
    storage:
      type: {{ .Values.kafka.cluster.storage.type }}
      size: {{ .Values.kafka.cluster.storage.size }}
      class: {{ .Values.kafka.cluster.storage.storageClass }}
      deleteClaim: false  # Keep data when deleting cluster

  # Zookeeper ensemble
  zookeeper:
    replicas: {{ .Values.kafka.zookeeper.replicas }}
    storage:
      type: {{ .Values.kafka.zookeeper.storage.type }}
      size: {{ .Values.kafka.zookeeper.storage.size }}
      class: {{ .Values.kafka.zookeeper.storage.storageClass }}
      deleteClaim: false

  # Entity Operator (manages topics and users)
  entityOperator:
    topicOperator: {}
    userOperator: {}
```

Kafka topics for our event-driven architecture (`templates/kafka-topics.yaml`):

```yaml
# Query flow topics
---
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: query.submitted
  namespace: {{ .Values.global.namespace }}
  labels:
    strimzi.io/cluster: {{ .Values.kafka.cluster.name }}
spec:
  partitions: 3
  replicas: 3
  config:
    retention.ms: 604800000  # 7 days
    segment.bytes: 1073741824  # 1GB
    compression.type: producer

---
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: data.collection.started
  namespace: {{ .Values.global.namespace }}
  labels:
    strimzi.io/cluster: {{ .Values.kafka.cluster.name }}
spec:
  partitions: 3
  replicas: 3
  config:
    retention.ms: 604800000
    segment.bytes: 1073741824
    compression.type: producer

---
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: data.collection.completed
  namespace: {{ .Values.global.namespace }}
  labels:
    strimzi.io/cluster: {{ .Values.kafka.cluster.name }}
spec:
  partitions: 3
  replicas: 3
  config:
    retention.ms: 604800000
    segment.bytes: 1073741824
    compression.type: producer

# ... (16 total topics for complete event flow)
```

**Topics I created:**

1. **Query Topics**: `query.submitted`, `query.validated`
2. **Data Collection**: `data.collection.{started,completed,failed}`
3. **Graph Processing**: `graph.processing.{started,completed,failed}`
4. **Vector Processing**: `vector.processing.{started,completed,failed}`
5. **Reasoning**: `reasoning.{started,completed,failed}`
6. **Health**: `agent.health.check`, `agent.error`

### Step 6: Neo4j StatefulSet

Neo4j needs persistent storage and stable network identity:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {{ include "researcherai.fullname" . }}-neo4j
  namespace: {{ .Values.global.namespace }}
spec:
  serviceName: {{ include "researcherai.fullname" . }}-neo4j
  replicas: 1  # Single instance for simplicity

  selector:
    matchLabels:
      app: neo4j

  template:
    metadata:
      labels:
        app: neo4j
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 7474
        fsGroup: 7474

      containers:
        - name: neo4j
          image: neo4j:{{ .Values.neo4j.image.tag }}

          ports:
            - name: http
              containerPort: 7474
            - name: bolt
              containerPort: 7687

          env:
            # Memory configuration
            - name: NEO4J_dbms_memory_heap_initial__size
              value: {{ .Values.neo4j.config.dbms.memory.heap.initial_size }}
            - name: NEO4J_dbms_memory_heap_max__size
              value: {{ .Values.neo4j.config.dbms.memory.heap.max_size }}
            - name: NEO4J_dbms_memory_pagecache_size
              value: {{ .Values.neo4j.config.dbms.memory.pagecache.size }}

            # Enable APOC
            - name: NEO4J_apoc_export_file_enabled
              value: "true"
            - name: NEO4J_apoc_import_file_enabled
              value: "true"
            - name: NEO4J_dbms_security_procedures_unrestricted
              value: "apoc.*"

            # Authentication
            - name: NEO4J_AUTH
              valueFrom:
                secretKeyRef:
                  name: {{ include "researcherai.fullname" . }}-secrets
                  key: NEO4J_PASSWORD

          volumeMounts:
            - name: data
              mountPath: /data
            - name: logs
              mountPath: /logs

          resources:
            {{- toYaml .Values.neo4j.resources | nindent 12 }}

  # Volume claim templates
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: {{ .Values.neo4j.persistentVolume.storageClass }}
        resources:
          requests:
            storage: {{ .Values.neo4j.persistentVolume.size }}

    - metadata:
        name: logs
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: {{ .Values.neo4j.persistentVolume.storageClass }}
        resources:
          requests:
            storage: 1Gi
```

**Why StatefulSet?**

- **Stable network identity**: Pod always gets same DNS name
- **Ordered deployment**: Pods start/stop in order
- **Persistent storage**: Each pod gets its own persistent volume
- **Data survives restarts**: Storage persists even if pod is deleted

### Step 7: Service Discovery

Services make pods discoverable (`templates/service.yaml`):

```yaml
# Application service
apiVersion: v1
kind: Service
metadata:
  name: {{ include "researcherai.fullname" . }}
  namespace: {{ .Values.global.namespace }}
spec:
  type: ClusterIP
  ports:
    - port: 8000
      targetPort: http
      protocol: TCP
      name: http
  selector:
    {{- include "researcherai.selectorLabels" . | nindent 4 }}

---
# Neo4j service
apiVersion: v1
kind: Service
metadata:
  name: {{ include "researcherai.fullname" . }}-neo4j
  namespace: {{ .Values.global.namespace }}
spec:
  type: ClusterIP
  ports:
    - port: 7474
      targetPort: 7474
      name: http
    - port: 7687
      targetPort: 7687
      name: bolt
  selector:
    app: neo4j

---
# Qdrant service
apiVersion: v1
kind: Service
metadata:
  name: {{ include "researcherai.fullname" . }}-qdrant
  namespace: {{ .Values.global.namespace }}
spec:
  type: ClusterIP
  ports:
    - port: 6333
      targetPort: 6333
      name: http
    - port: 6334
      targetPort: 6334
      name: grpc
  selector:
    app.kubernetes.io/name: qdrant
```

**DNS names applications use:**

```python
# Automatic service discovery in Kubernetes
NEO4J_URI = "bolt://researcherai-neo4j:7687"
QDRANT_HOST = "researcherai-qdrant:6333"
KAFKA_BOOTSTRAP_SERVERS = "rag-kafka-kafka-bootstrap:9092"
```

### Step 8: Ingress for External Access

Ingress controller routes traffic from internet to our application:

```yaml
{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ include "researcherai.fullname" . }}
  namespace: {{ .Values.global.namespace }}
  annotations:
    # NGINX Ingress Controller
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"

    # cert-manager for Let's Encrypt SSL
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx

  tls:
    - hosts:
        - {{ .Values.ingress.host }}
      secretName: {{ include "researcherai.fullname" . }}-tls

  rules:
    - host: {{ .Values.ingress.host }}
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: {{ include "researcherai.fullname" . }}
                port:
                  number: 8000
{{- end }}
```

**Traffic flow:**

```
Internet
  ↓
Ingress Controller (nginx) → SSL termination
  ↓
Service (load balancer)
  ↓
Pods (2-10 replicas)
```

## Deploying to Kubernetes

Now let's deploy everything!

### Prerequisites

```bash
# 1. Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# 2. Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# 3. Verify installations
kubectl version --client
helm version
```

### Option 1: Automated Deployment Script

I created a deployment script that handles everything:

```bash
#!/bin/bash
# k8s/scripts/deploy.sh

set -e

echo "ResearcherAI Kubernetes Deployment"
echo "=================================="

# Check prerequisites
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl not found"
    exit 1
fi

if ! command -v helm &> /dev/null; then
    echo "Error: helm not found"
    exit 1
fi

# Check cluster connection
echo "Checking Kubernetes cluster connection..."
if ! kubectl cluster-info &> /dev/null; then
    echo "Error: Cannot connect to Kubernetes cluster"
    exit 1
fi

# Install Strimzi Kafka Operator
echo "Installing Strimzi Kafka Operator..."
kubectl create namespace kafka --dry-run=client -o yaml | kubectl apply -f -
helm repo add strimzi https://strimzi.io/charts/
helm repo update

helm upgrade --install strimzi-kafka-operator strimzi/strimzi-kafka-operator \
  --namespace kafka \
  --set watchAnyNamespace=true \
  --wait

# Get configuration
read -p "Enter your Google API key: " -s GOOGLE_API_KEY
echo
read -p "Enter Neo4j password: " -s NEO4J_PASSWORD
echo

# Create values file
cat > /tmp/researcherai-values.yaml <<EOF
app:
  secrets:
    googleApiKey: "${GOOGLE_API_KEY}"
    neo4jPassword: "${NEO4J_PASSWORD}"

  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10

neo4j:
  enabled: true

qdrant:
  enabled: true

kafka:
  enabled: true
EOF

# Install/upgrade ResearcherAI
echo "Deploying ResearcherAI..."
helm upgrade --install researcherai ./k8s/helm/researcherai \
  --namespace researcherai \
  --create-namespace \
  --values /tmp/researcherai-values.yaml \
  --wait \
  --timeout 10m

# Clean up values file
rm /tmp/researcherai-values.yaml

# Show deployment status
echo ""
echo "Deployment completed successfully!"
echo ""
echo "Checking pod status..."
kubectl get pods -n researcherai

echo ""
echo "Getting service information..."
kubectl get services -n researcherai

echo ""
echo "To watch pod status:"
echo "  kubectl get pods -n researcherai --watch"
echo ""
echo "To view logs:"
echo "  kubectl logs -n researcherai -l app.kubernetes.io/name=researcherai --follow"
echo ""
echo "To access the application:"
echo "  kubectl port-forward -n researcherai svc/researcherai 8000:8000"
echo "  Then visit: http://localhost:8000"
```

**Run the deployment:**

```bash
chmod +x k8s/scripts/deploy.sh
./k8s/scripts/deploy.sh
```

### Option 2: Manual Step-by-Step Deployment

**Step 1: Install Strimzi Operator**

```bash
# Create Kafka namespace
kubectl create namespace kafka

# Add Strimzi Helm repo
helm repo add strimzi https://strimzi.io/charts/
helm repo update

# Install operator
helm install strimzi-kafka-operator strimzi/strimzi-kafka-operator \
  --namespace kafka \
  --set watchAnyNamespace=true
```

**Step 2: Create Configuration File**

```bash
# custom-values.yaml
app:
  replicaCount: 2
  secrets:
    googleApiKey: "YOUR_GOOGLE_API_KEY"
    neo4jPassword: "secure-neo4j-password"

  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 80

neo4j:
  enabled: true
  config:
    dbms.memory.heap.initial_size: "2G"
    dbms.memory.heap.max_size: "2G"

qdrant:
  enabled: true
  persistence:
    size: 20Gi  # Adjust based on your needs

kafka:
  enabled: true
  cluster:
    replicas: 3

ingress:
  enabled: true
  host: researcherai.yourdomain.com
```

**Step 3: Install ResearcherAI**

```bash
helm install researcherai ./k8s/helm/researcherai \
  --namespace researcherai \
  --create-namespace \
  --values custom-values.yaml \
  --wait
```

**Step 4: Verify Deployment**

```bash
# Watch pods start up
kubectl get pods -n researcherai --watch

# Expected output:
NAME                                   READY   STATUS    RESTARTS   AGE
researcherai-6b8f7d9c5d-abcde         1/1     Running   0          2m
researcherai-6b8f7d9c5d-fghij         1/1     Running   0          2m
researcherai-neo4j-0                  1/1     Running   0          3m
researcherai-qdrant-7d9b8c6f5-xyz     1/1     Running   0          3m
rag-kafka-kafka-0                      1/1     Running   0          4m
rag-kafka-kafka-1                      1/1     Running   0          4m
rag-kafka-kafka-2                      1/1     Running   0          4m
rag-kafka-zookeeper-0                  1/1     Running   0          5m
rag-kafka-zookeeper-1                  1/1     Running   0          5m
rag-kafka-zookeeper-2                  1/1     Running   0          5m

# Check services
kubectl get services -n researcherai

# Check Kafka topics
kubectl get kafkatopics -n researcherai
```

**Step 5: Access the Application**

```bash
# Port forward to local machine
kubectl port-forward -n researcherai svc/researcherai 8000:8000

# Visit in browser
open http://localhost:8000
```

## Testing Auto-Scaling

Let's verify that auto-scaling works:

### Generate Load

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Generate load (100 concurrent requests, 10000 total)
ab -n 10000 -c 100 http://localhost:8000/api/query
```

### Watch Scaling

```bash
# In another terminal, watch HPA
kubectl get hpa -n researcherai --watch

# Expected output:
NAME           REFERENCE                 TARGETS          MINPODS   MAXPODS   REPLICAS   AGE
researcherai   Deployment/researcherai   45%/80%, 30%/80%   2        10        2         10m
researcherai   Deployment/researcherai   85%/80%, 35%/80%   2        10        3         10m
researcherai   Deployment/researcherai   92%/80%, 40%/80%   2        10        4         11m

# Watch pods being created
kubectl get pods -n researcherai --watch
```

## Monitoring Deployment

### View Logs

```bash
# All application pods
kubectl logs -n researcherai -l app.kubernetes.io/name=researcherai --follow

# Specific pod
kubectl logs -n researcherai researcherai-6b8f7d9c5d-abcde --follow

# Previous crashed container
kubectl logs -n researcherai researcherai-6b8f7d9c5d-abcde --previous

# Multiple containers in a pod
kubectl logs -n researcherai researcherai-6b8f7d9c5d-abcde -c researcherai
```

### Check Resource Usage

```bash
# Pod resource usage
kubectl top pods -n researcherai

# Node resource usage
kubectl top nodes

# Detailed pod description
kubectl describe pod -n researcherai researcherai-6b8f7d9c5d-abcde
```

### Check Events

```bash
# Namespace events
kubectl get events -n researcherai --sort-by='.lastTimestamp'

# Watch events in real-time
kubectl get events -n researcherai --watch
```

## Updating the Application

### Rolling Update

```bash
# Update image tag in values
app:
  image:
    tag: "2.1.0"  # New version

# Apply upgrade
helm upgrade researcherai ./k8s/helm/researcherai \
  --namespace researcherai \
  --values custom-values.yaml

# Watch rolling update
kubectl rollout status deployment/researcherai -n researcherai
```

### Rollback if Needed

```bash
# View revision history
helm history researcherai -n researcherai

# Rollback to previous revision
helm rollback researcherai -n researcherai

# Rollback to specific revision
helm rollback researcherai 3 -n researcherai
```

## Production Considerations

### 1. Resource Planning

**Calculate Total Resources:**

```yaml
# Application: 2-10 pods
CPU: 1000m request × 2 pods = 2 cores minimum
CPU: 2000m limit × 10 pods = 20 cores maximum
RAM: 2Gi × 2 = 4Gi minimum
RAM: 4Gi × 10 = 40Gi maximum

# Neo4j: 1 pod
CPU: 1 core
RAM: 4Gi (2G heap + 1G page cache + overhead)
Storage: 10Gi

# Qdrant: 1 pod
CPU: 1 core
RAM: 2Gi
Storage: 10Gi

# Kafka: 3 brokers
CPU: 3 cores
RAM: 6Gi
Storage: 30Gi

# Zookeeper: 3 nodes
CPU: 1.5 cores
RAM: 3Gi
Storage: 15Gi

# TOTAL MINIMUM:
CPU: ~10 cores
RAM: ~19Gi
Storage: ~65Gi

# TOTAL MAXIMUM (when auto-scaled):
CPU: ~30 cores
RAM: ~55Gi
```

**Kubernetes Cluster Sizing:**

For production, I recommend:
- **3-node cluster** (high availability)
- **Each node**: 8 cores, 32GB RAM, 100GB SSD
- **Total**: 24 cores, 96GB RAM, 300GB storage
- **Cost**: ~$500-700/month (varies by cloud provider)

### 2. High Availability Checklist

- ✅ Multiple pod replicas (2-10)
- ✅ Pod Disruption Budget configured
- ✅ Kafka 3-broker cluster
- ✅ Zookeeper 3-node ensemble
- ✅ Rolling updates with maxUnavailable: 0
- ⚠️ Neo4j single replica (consider clustering)
- ⚠️ Qdrant single replica (configure replication)

### 3. Backup Strategy

**What to backup:**
- Neo4j data (persistent volume)
- Qdrant collections (persistent volume)
- Kubernetes configurations (Git)
- Secrets (external secret management)

**Backup tools:**

```bash
# Install Velero for Kubernetes backups
helm repo add vmware-tanzu https://vmware-tanzu.github.io/helm-charts
helm install velero vmware-tanzu/velero \
  --namespace velero \
  --create-namespace \
  --set configuration.provider=aws \
  --set configuration.backupStorageLocation.bucket=researcherai-backups \
  --set configuration.backupStorageLocation.config.region=us-east-1

# Schedule daily backups
velero schedule create daily-backup \
  --schedule="0 2 * * *" \
  --include-namespaces researcherai
```

### 4. Security Hardening

**Network Policies:**

```yaml
# Restrict pod-to-pod communication
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: researcherai-network-policy
  namespace: researcherai
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: researcherai

  policyTypes:
    - Ingress
    - Egress

  ingress:
    # Allow from ingress controller
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000

  egress:
    # Allow to Neo4j
    - to:
        - podSelector:
            matchLabels:
              app: neo4j
      ports:
        - protocol: TCP
          port: 7687

    # Allow to Qdrant
    - to:
        - podSelector:
            matchLabels:
              app.kubernetes.io/name: qdrant
      ports:
        - protocol: TCP
          port: 6333

    # Allow to Kafka
    - to:
        - podSelector:
            matchLabels:
              strimzi.io/cluster: rag-kafka
      ports:
        - protocol: TCP
          port: 9092

    # Allow DNS
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53

    # Allow internet (for API calls)
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 443
```

**Pod Security Standards:**

```yaml
# Enforce restricted security standard
apiVersion: v1
kind: Namespace
metadata:
  name: researcherai
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

### 5. Cost Optimization

**Tips to reduce costs:**

1. **Use spot/preemptible instances** for non-production
2. **Right-size resources** - adjust requests/limits based on actual usage
3. **Scale to zero** for dev/staging environments when not in use
4. **Use cluster autoscaler** - add/remove nodes based on demand
5. **Monitor costs** - use tools like Kubecost

**Example: Development Environment**

```yaml
# dev-values.yaml
app:
  replicaCount: 1  # Only 1 replica
  resources:
    requests:
      cpu: 500m     # Half resources
      memory: 1Gi
  autoscaling:
    enabled: false  # No auto-scaling in dev

neo4j:
  config:
    dbms.memory.heap.initial_size: "512M"  # Less memory
    dbms.memory.heap.max_size: "512M"

kafka:
  cluster:
    replicas: 1  # Single broker for dev
  zookeeper:
    replicas: 1  # Single zookeeper
```

## Troubleshooting

### Pod Won't Start

```bash
# Check pod status
kubectl describe pod -n researcherai <pod-name>

# Common issues:
# 1. Image pull error - check image name/tag
# 2. Init container failing - check dependency services
# 3. Resource constraints - check node resources

# Check events
kubectl get events -n researcherai --sort-by='.lastTimestamp'
```

### Application Crashes

```bash
# Check logs
kubectl logs -n researcherai <pod-name> --previous

# Check resource limits
kubectl top pod -n researcherai <pod-name>

# Common issues:
# - Out of memory (OOMKilled)
# - Unhandled exceptions
# - Database connection failures
```

### Can't Connect to Database

```bash
# Test Neo4j connection
kubectl run -it --rm debug --image=busybox --restart=Never -n researcherai -- sh
nc -zv researcherai-neo4j 7687

# Check Neo4j logs
kubectl logs -n researcherai researcherai-neo4j-0

# Verify service
kubectl get svc -n researcherai researcherai-neo4j
```

### Kafka Issues

```bash
# Check Kafka cluster status
kubectl get kafka -n researcherai

# Check topics
kubectl get kafkatopics -n researcherai

# Connect to Kafka pod
kubectl exec -it -n researcherai rag-kafka-kafka-0 -- bash

# Inside pod, test Kafka
kafka-topics.sh --bootstrap-server localhost:9092 --list
```

## Next Steps

We've successfully deployed ResearcherAI to Kubernetes with:
- ✅ Auto-scaling (2-10 replicas)
- ✅ High availability (multiple brokers, rolling updates)
- ✅ Persistent storage (StatefulSets, PVCs)
- ✅ Service discovery (Kubernetes DNS)
- ✅ Event streaming (Kafka cluster)
- ✅ Health checks (liveness/readiness probes)

But we're not done yet! In the next chapter, we'll cover:
- **Terraform** - Infrastructure as Code for cloud resources
- **Observability** - Prometheus, Grafana, distributed tracing
- **CI/CD** - Automated deployments with GitHub Actions
- **Production readiness** - Security, backup, disaster recovery

Let's continue with Terraform deployment in the next chapter!
