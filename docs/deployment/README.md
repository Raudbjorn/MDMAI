# TTRPG MCP Server - Complete Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Environment Requirements](#environment-requirements)
3. [Docker Deployment](#docker-deployment)
4. [Systemd Service Deployment](#systemd-service-deployment)
5. [Cloud Platform Deployments](#cloud-platform-deployments)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Development Environment Setup](#development-environment-setup)
8. [Configuration Management](#configuration-management)
9. [Security Hardening](#security-hardening)
10. [Monitoring & Observability](#monitoring--observability)
11. [Backup & Recovery](#backup--recovery)
12. [Troubleshooting](#troubleshooting)
13. [Performance Tuning](#performance-tuning)
14. [Maintenance & Updates](#maintenance--updates)

## Overview

The TTRPG MCP Server can be deployed in various configurations to suit different needs:

- **Development**: Local setup for development and testing
- **Production**: High-availability deployment with monitoring
- **Cloud**: Scalable deployment on cloud platforms
- **Containerized**: Docker and Kubernetes deployments
- **Hybrid**: On-premises with cloud integration

## Environment Requirements

### System Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU** | 2 cores | 4+ cores | More cores improve PDF processing |
| **RAM** | 4 GB | 8+ GB | ChromaDB and embeddings are memory-intensive |
| **Storage** | 10 GB | 50+ GB | Depends on number of PDFs and cache size |
| **Network** | 10 Mbps | 100+ Mbps | For downloading models and serving content |

### Operating System Support

| OS | Support Level | Notes |
|----|---------------|--------|
| **Ubuntu 20.04/22.04** | ✅ Fully Supported | Recommended for production |
| **CentOS 8/Rocky Linux 8** | ✅ Fully Supported | Enterprise environments |
| **Debian 11/12** | ✅ Fully Supported | Stable and reliable |
| **macOS 11+** | ✅ Development Only | Not recommended for production |
| **Windows 10/11** | ⚠️ Limited Support | WSL recommended |

### Software Dependencies

**Core Requirements:**
```bash
# System packages
python3.11+
python3-pip
python3-venv
git
curl
build-essential  # Linux
```

**Optional Components:**
```bash
# GPU Support
nvidia-docker2      # For GPU containers
cuda-toolkit-11.8   # CUDA support
```

**Database Options:**
```bash
# Vector Database (Choose one)
chromadb           # Default, lightweight
qdrant            # High-performance alternative
weaviate          # Enterprise option

# Cache (Optional)
redis-server      # Performance boost
memcached         # Alternative caching
```

## Docker Deployment

### Quick Start

```bash
# Clone repository
git clone https://github.com/Raudbjorn/MDMAI.git
cd MDMAI

# Build and run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f ttrpg-assistant
```

### Production Docker Setup

#### 1. Multi-stage Dockerfile

```dockerfile
# /home/svnbjrn/code/phase12/MDMAI/Dockerfile.prod
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r ttrpg && useradd -r -g ttrpg ttrpg

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application
WORKDIR /app
COPY --chown=ttrpg:ttrpg . .

# Create data directories
RUN mkdir -p /data/{chromadb,cache,logs} && \
    chown -R ttrpg:ttrpg /data

# Switch to non-root user
USER ttrpg

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Default command
CMD ["python", "src/main.py"]
```

#### 2. Docker Compose Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  ttrpg-assistant:
    build:
      context: .
      dockerfile: Dockerfile.prod
    container_name: ttrpg-assistant
    restart: unless-stopped
    ports:
      - "8000:8000"
      - "9090:9090"
    volumes:
      - ttrpg_data:/data
      - ttrpg_config:/config:ro
    environment:
      - CHROMA_DB_PATH=/data/chromadb
      - CACHE_DIR=/data/cache
      - LOG_LEVEL=INFO
      - ENABLE_METRICS=true
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    depends_on:
      - redis
    networks:
      - ttrpg-network

  redis:
    image: redis:7-alpine
    container_name: ttrpg-redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - ttrpg-network

  nginx:
    image: nginx:alpine
    container_name: ttrpg-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - ttrpg_ssl:/etc/ssl/certs:ro
    depends_on:
      - ttrpg-assistant
    networks:
      - ttrpg-network

volumes:
  ttrpg_data:
    driver: local
  ttrpg_config:
    driver: local
  redis_data:
    driver: local
  ttrpg_ssl:
    driver: local

networks:
  ttrpg-network:
    driver: bridge
```

#### 3. GPU-Enabled Docker

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-venv \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Continue with standard setup...
# (Copy from Dockerfile.prod)

# GPU-specific environment
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

```bash
# Build and run with GPU support
docker build -f Dockerfile.gpu -t ttrpg-assistant:gpu .
docker run --gpus all -d ttrpg-assistant:gpu
```

### Docker Management Commands

```bash
# Build and deploy
make docker-build
make docker-deploy

# View logs
docker-compose logs -f ttrpg-assistant

# Scale services
docker-compose up --scale ttrpg-assistant=3

# Update deployment
docker-compose pull
docker-compose up -d

# Backup data
docker run --rm -v ttrpg_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/ttrpg-data-$(date +%Y%m%d).tar.gz /data

# Restore data
docker run --rm -v ttrpg_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/ttrpg-data-20240115.tar.gz -C /
```

## Systemd Service Deployment

### Installation Script

```bash
#!/bin/bash
# deploy/scripts/install-systemd.sh

set -euo pipefail

# Configuration
INSTALL_USER="ttrpg"
INSTALL_DIR="/opt/ttrpg-assistant"
DATA_DIR="/var/lib/ttrpg-assistant"
CONFIG_DIR="/etc/ttrpg-assistant"
SERVICE_NAME="ttrpg-assistant"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root"
   exit 1
fi

echo "Installing TTRPG Assistant as systemd service..."

# Create user
if ! id "$INSTALL_USER" &>/dev/null; then
    useradd -r -s /bin/false -d "$INSTALL_DIR" "$INSTALL_USER"
    echo "Created user: $INSTALL_USER"
fi

# Create directories
mkdir -p "$INSTALL_DIR" "$DATA_DIR" "$CONFIG_DIR"
chown "$INSTALL_USER:$INSTALL_USER" "$INSTALL_DIR" "$DATA_DIR"

# Copy application files
rsync -av --chown="$INSTALL_USER:$INSTALL_USER" \
    src/ requirements.txt config/ "$INSTALL_DIR/"

# Install Python dependencies
cd "$INSTALL_DIR"
sudo -u "$INSTALL_USER" python -m venv venv
sudo -u "$INSTALL_USER" venv/bin/pip install -r requirements.txt

# Create configuration files
cat > "$CONFIG_DIR/environment" << EOF
CHROMA_DB_PATH=$DATA_DIR/chromadb
CACHE_DIR=$DATA_DIR/cache
LOG_LEVEL=INFO
ENABLE_METRICS=true
METRICS_PORT=9090
EOF

# Create systemd service
cat > "/etc/systemd/system/$SERVICE_NAME.service" << EOF
[Unit]
Description=TTRPG Assistant MCP Server
After=network.target
Wants=network.target

[Service]
Type=simple
User=$INSTALL_USER
Group=$INSTALL_USER
WorkingDirectory=$INSTALL_DIR
Environment=PYTHONPATH=$INSTALL_DIR
EnvironmentFile=$CONFIG_DIR/environment
ExecStart=$INSTALL_DIR/venv/bin/python src/main.py
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$DATA_DIR $CONFIG_DIR
CapabilityBoundingSet=

# Resource limits
LimitNOFILE=65535
MemoryMax=2G

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl start "$SERVICE_NAME"

echo "Installation complete!"
echo "Service status: systemctl status $SERVICE_NAME"
echo "Logs: journalctl -u $SERVICE_NAME -f"
```

### Service Management

```bash
# Install service
sudo bash deploy/scripts/install-systemd.sh

# Control service
sudo systemctl start ttrpg-assistant
sudo systemctl stop ttrpg-assistant  
sudo systemctl restart ttrpg-assistant
sudo systemctl reload ttrpg-assistant

# Check status
sudo systemctl status ttrpg-assistant
sudo journalctl -u ttrpg-assistant -f

# Enable/disable automatic startup
sudo systemctl enable ttrpg-assistant
sudo systemctl disable ttrpg-assistant
```

### Service Configuration

```ini
# /etc/systemd/system/ttrpg-assistant.service
[Unit]
Description=TTRPG Assistant MCP Server
Documentation=https://github.com/Raudbjorn/MDMAI
After=network-online.target
Wants=network-online.target
RequiresMountsFor=/var/lib/ttrpg-assistant

[Service]
Type=notify
User=ttrpg
Group=ttrpg
WorkingDirectory=/opt/ttrpg-assistant
Environment=PYTHONPATH=/opt/ttrpg-assistant
EnvironmentFile=/etc/ttrpg-assistant/environment

# Main process
ExecStart=/opt/ttrpg-assistant/venv/bin/python src/main.py
ExecReload=/bin/kill -HUP $MAINPID

# Restart policy
Restart=always
RestartSec=5
TimeoutStartSec=60
TimeoutStopSec=30

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ttrpg-assistant

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/ttrpg-assistant /etc/ttrpg-assistant
CapabilityBoundingSet=
SystemCallFilter=@system-service
SystemCallErrorNumber=EPERM

# Resource limits
LimitNOFILE=65535
LimitNPROC=4096
MemoryMax=4G
TasksMax=4096

[Install]
WantedBy=multi-user.target
Alias=ttrpg.service
```

## Cloud Platform Deployments

### AWS Deployment

#### 1. EC2 Instance Setup

```bash
# Launch EC2 instance (t3.medium or larger)
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type t3.medium \
  --key-name my-key \
  --security-group-ids sg-12345678 \
  --subnet-id subnet-12345678 \
  --user-data file://deploy/cloud/aws-userdata.sh

# User data script
cat > deploy/cloud/aws-userdata.sh << 'EOF'
#!/bin/bash
yum update -y
yum install -y docker git

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Start Docker
systemctl start docker
systemctl enable docker

# Clone and deploy
git clone https://github.com/Raudbjorn/MDMAI.git /opt/ttrpg-assistant
cd /opt/ttrpg-assistant
docker-compose -f docker-compose.prod.yml up -d
EOF
```

#### 2. ECS Deployment

```yaml
# deploy/cloud/aws-ecs-task.yml
family: ttrpg-assistant
networkMode: awsvpc
requiresCompatibilities:
  - FARGATE
cpu: 1024
memory: 2048
executionRoleArn: arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole

containerDefinitions:
  - name: ttrpg-assistant
    image: your-registry/ttrpg-assistant:latest
    portMappings:
      - containerPort: 8000
        protocol: tcp
    environment:
      - name: CHROMA_DB_PATH
        value: /data/chromadb
      - name: LOG_LEVEL
        value: INFO
    mountPoints:
      - sourceVolume: ttrpg-data
        containerPath: /data
    logConfiguration:
      logDriver: awslogs
      options:
        awslogs-group: /ecs/ttrpg-assistant
        awslogs-region: us-east-1
        awslogs-stream-prefix: ecs

volumes:
  - name: ttrpg-data
    efsVolumeConfiguration:
      fileSystemId: fs-12345678
      rootDirectory: /
```

#### 3. EKS Kubernetes Deployment

```yaml
# deploy/cloud/aws-eks-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ttrpg-assistant
  namespace: ttrpg
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ttrpg-assistant
  template:
    metadata:
      labels:
        app: ttrpg-assistant
    spec:
      containers:
      - name: ttrpg-assistant
        image: your-registry/ttrpg-assistant:latest
        ports:
        - containerPort: 8000
        env:
        - name: CHROMA_DB_PATH
          value: /data/chromadb
        volumeMounts:
        - name: ttrpg-data
          mountPath: /data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: ttrpg-data
        persistentVolumeClaim:
          claimName: ttrpg-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ttrpg-assistant-service
spec:
  selector:
    app: ttrpg-assistant
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Google Cloud Platform (GCP)

#### 1. Compute Engine Deployment

```bash
# Create instance with startup script
gcloud compute instances create ttrpg-assistant \
  --zone=us-central1-a \
  --machine-type=e2-medium \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --metadata-from-file startup-script=deploy/cloud/gcp-startup.sh

# Startup script
cat > deploy/cloud/gcp-startup.sh << 'EOF'
#!/bin/bash
apt-get update
apt-get install -y docker.io docker-compose git

# Clone and deploy
git clone https://github.com/Raudbjorn/MDMAI.git /opt/ttrpg-assistant
cd /opt/ttrpg-assistant
docker-compose up -d
EOF
```

#### 2. Cloud Run Deployment

```yaml
# deploy/cloud/gcp-cloudrun.yml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ttrpg-assistant
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "1000m"
    spec:
      containers:
      - image: gcr.io/PROJECT_ID/ttrpg-assistant:latest
        ports:
        - containerPort: 8000
        env:
        - name: CHROMA_DB_PATH
          value: /tmp/chromadb  # Note: ephemeral storage
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### Microsoft Azure

#### 1. Container Instances

```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group ttrpg-rg \
  --name ttrpg-assistant \
  --image your-registry/ttrpg-assistant:latest \
  --cpu 1 \
  --memory 2 \
  --ports 8000 \
  --environment-variables \
    CHROMA_DB_PATH=/data/chromadb \
    LOG_LEVEL=INFO \
  --azure-file-volume-account-name mystorageaccount \
  --azure-file-volume-account-key $STORAGE_KEY \
  --azure-file-volume-share-name ttrpgdata \
  --azure-file-volume-mount-path /data
```

#### 2. App Service Deployment

```yaml
# deploy/cloud/azure-webapp.yml
apiVersion: 2021-02-01
kind: Microsoft.Web/sites
properties:
  serverFarmId: /subscriptions/SUBSCRIPTION_ID/resourceGroups/ttrpg-rg/providers/Microsoft.Web/serverfarms/ttrpg-plan
  siteConfig:
    linuxFxVersion: DOCKER|your-registry/ttrpg-assistant:latest
    appSettings:
      - name: WEBSITES_ENABLE_APP_SERVICE_STORAGE
        value: "false"
      - name: CHROMA_DB_PATH
        value: /home/data/chromadb
      - name: LOG_LEVEL
        value: INFO
    connectionStrings: []
```

## Kubernetes Deployment

### Complete Kubernetes Manifests

#### 1. Namespace and ConfigMap

```yaml
# k8s/namespace.yml
apiVersion: v1
kind: Namespace
metadata:
  name: ttrpg
  labels:
    name: ttrpg
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ttrpg-config
  namespace: ttrpg
data:
  CHROMA_DB_PATH: "/data/chromadb"
  CACHE_DIR: "/data/cache"
  LOG_LEVEL: "INFO"
  ENABLE_METRICS: "true"
  METRICS_PORT: "9090"
```

#### 2. Persistent Storage

```yaml
# k8s/storage.yml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ttrpg-data-pvc
  namespace: ttrpg
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd  # Adjust based on your cluster
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ttrpg-cache-pvc
  namespace: ttrpg
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd
```

#### 3. Deployment

```yaml
# k8s/deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ttrpg-assistant
  namespace: ttrpg
  labels:
    app: ttrpg-assistant
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: ttrpg-assistant
  template:
    metadata:
      labels:
        app: ttrpg-assistant
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: ttrpg-assistant
        image: your-registry/ttrpg-assistant:v1.0.0
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        envFrom:
        - configMapRef:
            name: ttrpg-config
        volumeMounts:
        - name: ttrpg-data
          mountPath: /data
        - name: ttrpg-cache
          mountPath: /cache
        - name: tmp
          mountPath: /tmp
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
      volumes:
      - name: ttrpg-data
        persistentVolumeClaim:
          claimName: ttrpg-data-pvc
      - name: ttrpg-cache
        persistentVolumeClaim:
          claimName: ttrpg-cache-pvc
      - name: tmp
        emptyDir: {}
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - ttrpg-assistant
              topologyKey: kubernetes.io/hostname
```

#### 4. Service and Ingress

```yaml
# k8s/service.yml
apiVersion: v1
kind: Service
metadata:
  name: ttrpg-assistant-service
  namespace: ttrpg
  labels:
    app: ttrpg-assistant
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  - port: 9090
    targetPort: 9090
    protocol: TCP
    name: metrics
  selector:
    app: ttrpg-assistant
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ttrpg-assistant-ingress
  namespace: ttrpg
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - ttrpg.yourdomain.com
    secretName: ttrpg-tls
  rules:
  - host: ttrpg.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ttrpg-assistant-service
            port:
              number: 80
```

#### 5. Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ttrpg-assistant-hpa
  namespace: ttrpg
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ttrpg-assistant
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### Kubernetes Deployment Commands

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n ttrpg
kubectl get services -n ttrpg
kubectl get ingress -n ttrpg

# View logs
kubectl logs -f deployment/ttrpg-assistant -n ttrpg

# Scale deployment
kubectl scale deployment ttrpg-assistant --replicas=5 -n ttrpg

# Update deployment
kubectl set image deployment/ttrpg-assistant \
  ttrpg-assistant=your-registry/ttrpg-assistant:v1.1.0 -n ttrpg

# Rollback deployment
kubectl rollout undo deployment/ttrpg-assistant -n ttrpg
```

## Development Environment Setup

### Local Development

```bash
# 1. Clone and setup
git clone https://github.com/Raudbjorn/MDMAI.git
cd MDMAI

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements-dev.txt

# 4. Setup pre-commit hooks
pre-commit install

# 5. Run tests
pytest tests/

# 6. Start development server
python src/main.py --dev
```

### Development Docker Setup

```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  ttrpg-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
      - "5678:5678"  # Debug port
    volumes:
      - .:/app
      - dev_data:/data
    environment:
      - DEBUG=true
      - LOG_LEVEL=DEBUG
      - RELOAD=true
    command: python -m debugpy --listen 0.0.0.0:5678 --wait-for-client src/main.py
volumes:
  dev_data:
```

### IDE Configuration

#### VS Code Settings

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ]
}
```

#### Debugging Configuration

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "program": "src/main.py",
            "console": "integratedTerminal",
            "env": {
                "DEBUG": "true",
                "LOG_LEVEL": "DEBUG"
            }
        },
        {
            "name": "Python: Attach to Docker",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            }
        }
    ]
}
```

## Configuration Management

### Environment Variables

```bash
# Core Configuration
export MCP_SERVER_NAME="TTRPG Assistant"
export LOG_LEVEL="INFO"
export DEBUG="false"

# Database Configuration
export CHROMA_DB_PATH="/var/lib/ttrpg-assistant/chromadb"
export CHROMA_DB_IMPL="duckdb+parquet"  # or "clickhouse"
export CONNECTION_POOL_SIZE="20"

# Cache Configuration
export CACHE_DIR="/var/lib/ttrpg-assistant/cache"
export CACHE_TTL_SECONDS="3600"
export CACHE_MAX_MEMORY_MB="500"
export REDIS_URL="redis://localhost:6379/0"

# Security Configuration
export ENABLE_AUTHENTICATION="true"
export AUTH_SECRET_KEY="your-secret-key-here"
export SESSION_TIMEOUT_MINUTES="60"
export ENABLE_RATE_LIMITING="true"

# Performance Configuration
export MAX_WORKERS="4"
export EMBEDDING_BATCH_SIZE="32"
export ENABLE_GPU="true"
export GPU_TYPE="cuda"  # or "mps" for Apple Silicon

# Monitoring Configuration
export ENABLE_METRICS="true"
export METRICS_PORT="9090"
export ENABLE_TRACING="false"
export SENTRY_DSN="your-sentry-dsn"

# Feature Flags
export ENABLE_COLLABORATION="true"
export ENABLE_AI_PROVIDERS="true"
export ENABLE_PDF_OCR="false"
```

### Configuration Files

#### Main Configuration

```yaml
# config/config.yml
server:
  name: "TTRPG Assistant"
  host: "0.0.0.0"
  port: 8000
  workers: 4
  
database:
  chromadb:
    path: "/var/lib/ttrpg-assistant/chromadb"
    implementation: "duckdb+parquet"
  
cache:
    type: "redis"  # or "memory"
    redis_url: "redis://localhost:6379/0"
    ttl_seconds: 3600
    max_memory_mb: 500

security:
  authentication:
    enabled: true
    secret_key: "${AUTH_SECRET_KEY}"
    session_timeout_minutes: 60
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    
monitoring:
  metrics:
    enabled: true
    port: 9090
  logging:
    level: "INFO"
    format: "structured"
  
features:
  collaboration: true
  ai_providers: true
  pdf_ocr: false
```

#### Logging Configuration

```yaml
# config/logging.yml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  structured:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: structured
    stream: ext://sys.stdout
    
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: standard
    filename: /var/log/ttrpg-assistant/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    
  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: structured
    filename: /var/log/ttrpg-assistant/error.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  '':  # root logger
    level: INFO
    handlers: [console, file, error_file]
    propagate: false
    
  ttrpg:
    level: DEBUG
    handlers: [console, file]
    propagate: false
    
  uvicorn:
    level: INFO
    handlers: [console, file]
    propagate: false
```

## Security Hardening

### System Security

```bash
# Firewall Configuration
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8000/tcp  # API (restrict to specific IPs in production)

# Disable unused services
sudo systemctl disable apache2
sudo systemctl disable nginx  # if not using as reverse proxy
sudo systemctl disable postgresql  # if not using

# System hardening
echo 'net.ipv4.conf.all.log_martians = 1' >> /etc/sysctl.conf
echo 'net.ipv4.conf.all.send_redirects = 0' >> /etc/sysctl.conf
echo 'net.ipv4.conf.all.accept_redirects = 0' >> /etc/sysctl.conf
sysctl -p
```

### Application Security

#### 1. SSL/TLS Configuration

```nginx
# /etc/nginx/sites-available/ttrpg-assistant
server {
    listen 80;
    server_name ttrpg.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ttrpg.yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/ttrpg.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ttrpg.yourdomain.com/privkey.pem;
    
    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_timeout 10m;
    ssl_session_cache shared:SSL:10m;
    ssl_session_tickets off;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';";

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Rate limiting
        limit_req zone=api burst=20 nodelay;
    }
    
    # Websocket support
    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}

# Rate limiting
http {
    limit_req_zone $binary_remote_addr zone=api:10m rate=60r/m;
}
```

#### 2. Authentication & Authorization

```python
# Security configuration
SECURITY_CONFIG = {
    "auth": {
        "jwt_secret": os.environ["JWT_SECRET"],
        "jwt_algorithm": "HS256",
        "access_token_expire_minutes": 30,
        "refresh_token_expire_days": 7
    },
    "password": {
        "min_length": 12,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_numbers": True,
        "require_special": True
    },
    "session": {
        "secure": True,
        "httponly": True,
        "samesite": "strict",
        "max_age": 3600
    }
}
```

#### 3. Input Validation

```python
# Input sanitization
from pydantic import BaseModel, validator
import bleach

class SearchRequest(BaseModel):
    query: str
    max_results: int = 5
    
    @validator('query')
    def sanitize_query(cls, v):
        # Remove HTML/JavaScript
        clean_query = bleach.clean(v, tags=[], strip=True)
        # Length limit
        if len(clean_query) > 1000:
            raise ValueError("Query too long")
        return clean_query
    
    @validator('max_results')
    def validate_max_results(cls, v):
        if v < 1 or v > 100:
            raise ValueError("max_results must be between 1 and 100")
        return v
```

#### 4. Audit Logging

```python
# Audit logging configuration
AUDIT_CONFIG = {
    "enabled": True,
    "log_file": "/var/log/ttrpg-assistant/audit.log",
    "events": [
        "authentication",
        "authorization",
        "data_access",
        "configuration_change",
        "security_event"
    ],
    "retention_days": 90
}
```

## Monitoring & Observability

### Metrics Collection

#### 1. Application Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info

# Request metrics
REQUEST_COUNT = Counter(
    'ttrpg_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'ttrpg_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint']
)

# Tool metrics
TOOL_CALLS = Counter(
    'ttrpg_tool_calls_total',
    'Total tool calls',
    ['tool', 'status']
)

TOOL_DURATION = Histogram(
    'ttrpg_tool_duration_seconds',
    'Tool execution duration',
    ['tool']
)

# Resource metrics
ACTIVE_SESSIONS = Gauge(
    'ttrpg_active_sessions',
    'Number of active sessions'
)

DATABASE_SIZE = Gauge(
    'ttrpg_database_size_bytes',
    'Database size in bytes'
)

# Application info
APP_INFO = Info(
    'ttrpg_app_info',
    'Application information'
)
```

#### 2. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "ttrpg_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'ttrpg-assistant'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
    scrape_interval: 5s
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
```

#### 3. Alerting Rules

```yaml
# ttrpg_rules.yml
groups:
- name: ttrpg_assistant
  rules:
  - alert: HighErrorRate
    expr: rate(ttrpg_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High error rate detected
      
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(ttrpg_request_duration_seconds_bucket[5m])) > 1.0
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: High response time detected
      
  - alert: ServiceDown
    expr: up{job="ttrpg-assistant"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: TTRPG Assistant service is down
```

### Logging & Tracing

#### 1. Structured Logging

```python
# logging_config.py
import structlog

def configure_logging():
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
```

#### 2. Distributed Tracing

```python
# tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def configure_tracing():
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    return tracer
```

### Health Checks

```python
# health.py
from datetime import datetime, timedelta
import psutil

class HealthChecker:
    def __init__(self):
        self.start_time = datetime.now()
        
    async def check_health(self):
        """Comprehensive health check"""
        checks = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "checks": {}
        }
        
        # Database check
        try:
            # Test ChromaDB connection
            checks["checks"]["database"] = {
                "status": "healthy",
                "response_time_ms": 12
            }
        except Exception as e:
            checks["checks"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            checks["status"] = "unhealthy"
        
        # Memory check
        memory = psutil.virtual_memory()
        checks["checks"]["memory"] = {
            "status": "healthy" if memory.percent < 90 else "warning",
            "usage_percent": memory.percent,
            "available_mb": memory.available // (1024 * 1024)
        }
        
        # Disk check
        disk = psutil.disk_usage('/')
        checks["checks"]["disk"] = {
            "status": "healthy" if disk.percent < 90 else "warning",
            "usage_percent": disk.percent,
            "available_gb": disk.free // (1024 * 1024 * 1024)
        }
        
        return checks
```

## Backup & Recovery

### Automated Backup System

```bash
#!/bin/bash
# deploy/scripts/backup.sh

set -euo pipefail

# Configuration
BACKUP_DIR="/var/backups/ttrpg-assistant"
DATA_DIR="/var/lib/ttrpg-assistant"
CONFIG_DIR="/etc/ttrpg-assistant"
RETENTION_DAYS=30
S3_BUCKET="ttrpg-backups"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="ttrpg-backup-$TIMESTAMP"

echo "Starting backup: $BACKUP_NAME"

# Create backup archive
tar -czf "$BACKUP_DIR/$BACKUP_NAME.tar.gz" \
  -C / \
  --exclude="$DATA_DIR/cache/*" \
  --exclude="$DATA_DIR/logs/*" \
  "$DATA_DIR" \
  "$CONFIG_DIR"

# Encrypt backup
gpg --cipher-algo AES256 --compress-algo 1 --s2k-mode 3 \
  --s2k-digest-algo SHA512 --s2k-count 65536 \
  --symmetric --output "$BACKUP_DIR/$BACKUP_NAME.tar.gz.gpg" \
  "$BACKUP_DIR/$BACKUP_NAME.tar.gz"

# Remove unencrypted backup
rm "$BACKUP_DIR/$BACKUP_NAME.tar.gz"

# Upload to S3 (optional)
if command -v aws &> /dev/null; then
  aws s3 cp "$BACKUP_DIR/$BACKUP_NAME.tar.gz.gpg" \
    "s3://$S3_BUCKET/daily/$BACKUP_NAME.tar.gz.gpg"
fi

# Cleanup old backups
find "$BACKUP_DIR" -name "ttrpg-backup-*.tar.gz.gpg" \
  -mtime +$RETENTION_DAYS -delete

echo "Backup completed: $BACKUP_NAME"
```

### Recovery Procedures

```bash
#!/bin/bash
# deploy/scripts/restore.sh

set -euo pipefail

BACKUP_FILE="$1"
DATA_DIR="/var/lib/ttrpg-assistant"
CONFIG_DIR="/etc/ttrpg-assistant"
TEMP_DIR="/tmp/ttrpg-restore"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

echo "Starting restore from: $BACKUP_FILE"

# Stop service
systemctl stop ttrpg-assistant

# Create temp directory
mkdir -p "$TEMP_DIR"

# Decrypt and extract backup
gpg --decrypt "$BACKUP_FILE" | tar -xzf - -C "$TEMP_DIR"

# Backup current data
if [[ -d "$DATA_DIR" ]]; then
    mv "$DATA_DIR" "${DATA_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
fi

if [[ -d "$CONFIG_DIR" ]]; then
    mv "$CONFIG_DIR" "${CONFIG_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
fi

# Restore data
mv "$TEMP_DIR$DATA_DIR" "$DATA_DIR"
mv "$TEMP_DIR$CONFIG_DIR" "$CONFIG_DIR"

# Fix permissions
chown -R ttrpg:ttrpg "$DATA_DIR"
chmod -R 750 "$DATA_DIR"

# Start service
systemctl start ttrpg-assistant

# Cleanup
rm -rf "$TEMP_DIR"

echo "Restore completed successfully"
```

### Database-Specific Backups

```python
# backup_manager.py
import asyncio
import shutil
import gzip
from datetime import datetime
from pathlib import Path

class BackupManager:
    def __init__(self, data_dir: Path, backup_dir: Path):
        self.data_dir = data_dir
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_backup(self, compress: bool = True) -> Path:
        """Create a complete backup"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"ttrpg_backup_{timestamp}"
        
        # Create backup directory
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        # Backup ChromaDB
        chromadb_source = self.data_dir / "chromadb"
        chromadb_backup = backup_path / "chromadb"
        
        if chromadb_source.exists():
            shutil.copytree(chromadb_source, chromadb_backup)
        
        # Backup configuration
        config_source = Path("/etc/ttrpg-assistant")
        config_backup = backup_path / "config"
        
        if config_source.exists():
            shutil.copytree(config_source, config_backup)
        
        # Create manifest
        manifest = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": ["chromadb", "config"],
            "size_bytes": self._get_directory_size(backup_path)
        }
        
        with open(backup_path / "manifest.json", "w") as f:
            import json
            json.dump(manifest, f, indent=2)
        
        # Compress if requested
        if compress:
            archive_path = backup_path.with_suffix(".tar.gz")
            shutil.make_archive(str(backup_path), "gztar", str(backup_path))
            shutil.rmtree(backup_path)
            return archive_path
        
        return backup_path
    
    def _get_directory_size(self, path: Path) -> int:
        """Calculate directory size recursively"""
        total_size = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
```

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

```bash
# Check service status
systemctl status ttrpg-assistant

# Check logs
journalctl -u ttrpg-assistant -n 50

# Common fixes
sudo systemctl daemon-reload
sudo systemctl reset-failed ttrpg-assistant
sudo systemctl start ttrpg-assistant
```

**Typical causes:**
- Port already in use
- Permission issues
- Missing dependencies  
- Configuration errors
- Database corruption

#### 2. High Memory Usage

```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check ChromaDB size
du -sh /var/lib/ttrpg-assistant/chromadb

# Solutions
systemctl restart ttrpg-assistant  # Clear memory leaks
# Reduce cache sizes in configuration
# Add more RAM or enable swap
```

#### 3. Slow Response Times

```bash
# Check system load
top
iostat -x 1

# Check database performance
# Enable query logging in ChromaDB
export CHROMA_LOG_LEVEL=DEBUG

# Solutions
# Optimize database indexes
# Increase worker processes
# Enable caching
# Use SSD storage
```

#### 4. Database Connection Errors

```bash
# Check database files
ls -la /var/lib/ttrpg-assistant/chromadb/
lsof /var/lib/ttrpg-assistant/chromadb/chroma.sqlite3

# Check permissions
sudo -u ttrpg ls -la /var/lib/ttrpg-assistant/chromadb/

# Recovery
# Stop service, backup data, recreate database
systemctl stop ttrpg-assistant
cp -r /var/lib/ttrpg-assistant/chromadb /tmp/chromadb-backup
# Restore from backup if needed
```

### Debug Mode

```bash
# Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG
export SQL_ECHO=true

# Run with debugging
python -m pdb src/main.py

# Docker debug mode
docker run -it --rm \
  -e DEBUG=true \
  -e LOG_LEVEL=DEBUG \
  your-registry/ttrpg-assistant:latest \
  bash
```

### Performance Profiling

```python
# profile.py
import cProfile
import pstats
from src.main import main

def profile_application():
    """Profile the application startup and initial requests"""
    profiler = cProfile.Profile()
    
    profiler.enable()
    # Run application code
    main()
    profiler.disable()
    
    # Save results
    profiler.dump_stats('ttrpg_profile.stats')
    
    # Print top time consumers
    stats = pstats.Stats('ttrpg_profile.stats')
    stats.sort_stats('cumulative')
    stats.print_stats(20)

if __name__ == "__main__":
    profile_application()
```

## Performance Tuning

### Database Optimization

```python
# ChromaDB optimization
CHROMADB_CONFIG = {
    "implementation": "duckdb+parquet",  # Faster than SQLite
    "connection_pool_size": 20,
    "query_cache_size": 1000,
    "batch_size": 100,
    "parallel_workers": 4
}

# Embedding optimization
EMBEDDING_CONFIG = {
    "model": "all-MiniLM-L6-v2",  # Balance of speed and quality
    "batch_size": 64,  # Increase for better GPU utilization
    "max_seq_length": 512,
    "normalize_embeddings": True
}
```

### Caching Strategy

```python
# Multi-level caching
CACHE_CONFIG = {
    "l1_memory": {
        "type": "lru",
        "max_size": 1000,
        "ttl": 300  # 5 minutes
    },
    "l2_redis": {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "ttl": 3600,  # 1 hour
        "max_memory": "100mb"
    },
    "l3_disk": {
        "path": "/var/cache/ttrpg-assistant",
        "max_size_gb": 5,
        "ttl": 86400  # 24 hours
    }
}
```

### Connection Pooling

```python
# Async connection pooling
import asyncio
from contextlib import asynccontextmanager

class ConnectionPool:
    def __init__(self, max_connections=20):
        self.semaphore = asyncio.Semaphore(max_connections)
        self.connections = []
    
    @asynccontextmanager
    async def get_connection(self):
        async with self.semaphore:
            # Reuse existing connection or create new one
            connection = await self._get_or_create_connection()
            try:
                yield connection
            finally:
                await self._return_connection(connection)
```

### Resource Limits

```bash
# System limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf
echo "* soft nproc 4096" >> /etc/security/limits.conf
echo "* hard nproc 4096" >> /etc/security/limits.conf

# Memory limits
echo "vm.max_map_count = 262144" >> /etc/sysctl.conf
sysctl -p
```

## Maintenance & Updates

### Update Procedures

#### 1. Rolling Updates (Zero Downtime)

```bash
#!/bin/bash
# deploy/scripts/rolling-update.sh

set -euo pipefail

NEW_VERSION="$1"
CURRENT_VERSION=$(docker ps --format "table {{.Image}}" | grep ttrpg-assistant | awk -F: '{print $2}')

echo "Updating from $CURRENT_VERSION to $NEW_VERSION"

# Pull new image
docker pull your-registry/ttrpg-assistant:$NEW_VERSION

# Update docker-compose
sed -i "s/:$CURRENT_VERSION/:$NEW_VERSION/g" docker-compose.yml

# Rolling update
docker-compose up -d --no-deps ttrpg-assistant

# Health check
for i in {1..30}; do
    if curl -f http://localhost:8000/health; then
        echo "Health check passed"
        break
    fi
    echo "Waiting for service to be healthy..."
    sleep 10
done

echo "Update completed successfully"
```

#### 2. Database Migrations

```python
# migrate.py
from pathlib import Path
import json
import shutil
from datetime import datetime

class DatabaseMigrator:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.migrations_dir = Path(__file__).parent / "migrations"
    
    def get_current_version(self) -> str:
        """Get current database version"""
        version_file = self.data_dir / "version.json"
        if version_file.exists():
            with open(version_file) as f:
                return json.load(f)["version"]
        return "0.0.0"
    
    def apply_migrations(self, target_version: str):
        """Apply migrations to reach target version"""
        current = self.get_current_version()
        migrations = self._get_required_migrations(current, target_version)
        
        # Backup before migration
        backup_path = self._create_backup()
        
        try:
            for migration in migrations:
                print(f"Applying migration: {migration.name}")
                migration.apply(self.data_dir)
                self._update_version(migration.version)
                
        except Exception as e:
            print(f"Migration failed: {e}")
            print(f"Restoring from backup: {backup_path}")
            self._restore_backup(backup_path)
            raise
        
        print(f"Successfully migrated to version {target_version}")
```

### Scheduled Maintenance

```bash
# /etc/cron.d/ttrpg-maintenance
# Daily backup at 2 AM
0 2 * * * ttrpg /opt/ttrpg-assistant/deploy/scripts/backup.sh

# Weekly log rotation at 3 AM Sunday
0 3 * * 0 root logrotate -f /etc/logrotate.d/ttrpg-assistant

# Monthly cleanup at 4 AM on 1st
0 4 1 * * ttrpg /opt/ttrpg-assistant/deploy/scripts/cleanup.sh

# Health check every 5 minutes
*/5 * * * * root /opt/ttrpg-assistant/deploy/scripts/health-check.sh
```

### Cleanup Scripts

```bash
#!/bin/bash
# deploy/scripts/cleanup.sh

set -euo pipefail

echo "Starting maintenance cleanup..."

# Clean old logs
find /var/log/ttrpg-assistant -name "*.log.*" -mtime +30 -delete
find /var/log/ttrpg-assistant -name "*.gz" -mtime +30 -delete

# Clean cache
find /var/lib/ttrpg-assistant/cache -type f -mtime +7 -delete

# Clean temporary files
find /tmp -name "ttrpg-*" -mtime +1 -delete

# Compact database (if using SQLite)
sqlite3 /var/lib/ttrpg-assistant/chromadb/chroma.sqlite3 "VACUUM;"

# Clean Docker images
docker system prune -af --filter "until=720h"

echo "Cleanup completed"
```

This completes the comprehensive deployment guide covering all major deployment scenarios, configurations, and operational procedures for the TTRPG MCP Server.