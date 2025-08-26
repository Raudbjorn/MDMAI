# TTRPG Assistant MCP Server - Deployment Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation Methods](#installation-methods)
4. [Configuration](#configuration)
5. [Post-Installation](#post-installation)
6. [Troubleshooting](#troubleshooting)
7. [Security Considerations](#security-considerations)

## Overview

The TTRPG Assistant MCP Server can be deployed in multiple ways to suit different environments and requirements. This guide covers all supported deployment methods and best practices.

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4 GB
- Storage: 10 GB free space
- Python: 3.9 or higher
- Operating System: Linux, macOS, or Windows 10/11

**Recommended Requirements:**
- CPU: 4+ cores
- RAM: 8 GB
- Storage: 20 GB free space
- GPU: NVIDIA GPU with CUDA support (optional)
- Network: Stable internet connection for downloading models

### Software Dependencies

Required system packages:
- Python 3.9+
- pip or conda
- git
- curl or wget
- Build tools (gcc, make)

## Installation Methods

### 1. Standalone Installation (Recommended for Development)

#### Linux/macOS

```bash
# Clone the repository
git clone https://github.com/Raudbjorn/MDMAI.git
cd MDMAI

# Run the installation script
sudo bash deploy/scripts/install.sh \
  --install-dir /opt/ttrpg-assistant \
  --data-dir /var/lib/ttrpg-assistant \
  --mode standalone

# Or use the Makefile
make deploy-install
```

#### Windows

```powershell
# Clone the repository
git clone https://github.com/Raudbjorn/MDMAI.git
cd MDMAI

# Run the installation script (as Administrator)
.\deploy\scripts\install.ps1 `
  -InstallDir "C:\Program Files\TTRPG-Assistant" `
  -DataDir "C:\ProgramData\TTRPG-Assistant" `
  -InstallMode standalone
```

### 2. Docker Installation (Recommended for Production)

```bash
# Build the Docker image
docker build -t ttrpg-assistant:latest .

# Or use docker-compose
docker-compose up -d

# With GPU support (NVIDIA)
docker build --build-arg GPU_SUPPORT=cuda -t ttrpg-assistant:cuda .
docker run --gpus all -d ttrpg-assistant:cuda
```

### 3. Systemd Service Installation (Linux Production)

```bash
# Install as systemd service
sudo bash deploy/scripts/install.sh \
  --mode systemd \
  --user ttrpg

# Start the service
sudo systemctl start ttrpg-assistant
sudo systemctl enable ttrpg-assistant

# Check status
sudo systemctl status ttrpg-assistant
```

### 4. Kubernetes Deployment

```yaml
# ttrpg-assistant-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ttrpg-assistant
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
        image: ttrpg-assistant:latest
        ports:
        - containerPort: 8000
        env:
        - name: MCP_SERVER_NAME
          value: "TTRPG Assistant"
        volumeMounts:
        - name: data
          mountPath: /data
        - name: config
          mountPath: /config
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: ttrpg-data-pvc
      - name: config
        configMap:
          name: ttrpg-config
```

## Configuration

### Interactive Configuration Wizard

Run the configuration wizard for guided setup:

```bash
python deploy/scripts/configure.py --config-dir /etc/ttrpg-assistant

# Or use the Makefile
make deploy-configure
```

### Manual Configuration

1. Copy the configuration template:
```bash
cp deploy/config/.env.template /etc/ttrpg-assistant/.env
cp deploy/config/config.yaml.template /etc/ttrpg-assistant/config.yaml
```

2. Edit the configuration files:

**.env file:**
```bash
# Essential settings
MCP_SERVER_NAME=TTRPG Assistant
LOG_LEVEL=INFO
CHROMA_DB_PATH=/var/lib/ttrpg-assistant/chromadb
CACHE_DIR=/var/lib/ttrpg-assistant/cache

# Security
ENABLE_AUTHENTICATION=true
SESSION_TIMEOUT_MINUTES=60

# Performance
MAX_WORKERS=4
EMBEDDING_BATCH_SIZE=32

# GPU Support (if available)
ENABLE_GPU=true
GPU_TYPE=cuda
```

**config.yaml:**
```yaml
server:
  name: "TTRPG Assistant"
  host: "0.0.0.0"
  port: 8000
  
database:
  chromadb:
    path: "/var/lib/ttrpg-assistant/chromadb"
    
search:
  engine:
    type: "hybrid"
    default_limit: 5
```

### Environment-Specific Configurations

**Development:**
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
export HOT_RELOAD=true
```

**Production:**
```bash
export DEBUG=false
export LOG_LEVEL=WARNING
export ENABLE_AUTHENTICATION=true
export ENABLE_RATE_LIMITING=true
```

## Post-Installation

### 1. Verify Installation

```bash
# Check dependencies
python deploy/scripts/check_requirements.py

# Verify environment
python deploy/scripts/setup_environment.py

# Run health check
curl http://localhost:8000/health
```

### 2. Download Language Models

```bash
# Download spaCy models
python -m spacy download en_core_web_sm

# Download embedding models (automatic on first use)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### 3. Initialize Database

```bash
# Run initial migration
python deploy/migration/migrate.py 1.0.0

# Create initial backup
python deploy/backup/backup_manager.py --create
```

### 4. Load Initial Data

```bash
# Import rulebooks and sources
python -m src.source_management.mcp_tools add-source /path/to/rulebooks

# Create initial campaign
python -m src.campaign.mcp_tools create-campaign "My Campaign"
```

### 5. Configure Monitoring

```bash
# Enable metrics endpoint
export ENABLE_METRICS=true
export METRICS_PORT=9090

# Configure Prometheus scraping
# prometheus.yml
scrape_configs:
  - job_name: 'ttrpg-assistant'
    static_configs:
      - targets: ['localhost:9090']
```

## Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Find process using port
lsof -i :8000
# Or
netstat -tulpn | grep 8000

# Change port in configuration
export SERVER_PORT=8001
```

**2. Permission Denied**
```bash
# Fix permissions
sudo chown -R ttrpg:ttrpg /var/lib/ttrpg-assistant
sudo chmod 750 /var/lib/ttrpg-assistant
```

**3. Module Import Errors**
```bash
# Reinstall dependencies
pip install -e . --upgrade
# Or
make install-dev
```

**4. GPU Not Detected**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**5. Memory Issues**
```bash
# Increase memory limits
export CACHE_MAX_MEMORY_MB=500
export MAX_WORKERS=2

# For Docker
docker run -m 4g ttrpg-assistant:latest
```

### Logging

Check logs for detailed error information:

```bash
# Systemd logs
journalctl -u ttrpg-assistant -f

# Application logs
tail -f /var/log/ttrpg-assistant/ttrpg-assistant.log

# Security audit logs
tail -f /var/log/ttrpg-assistant/security.log

# Docker logs
docker logs -f ttrpg-assistant
```

### Debug Mode

Enable debug mode for detailed diagnostics:

```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
export SQL_ECHO=true

python -m src.main
```

## Security Considerations

### 1. Network Security

**Firewall Configuration:**
```bash
# Allow only necessary ports
sudo ufw allow 8000/tcp  # API port
sudo ufw allow 9090/tcp  # Metrics port (internal only)
```

**Reverse Proxy (Nginx):**
```nginx
server {
    listen 443 ssl http2;
    server_name ttrpg.example.com;
    
    ssl_certificate /etc/ssl/certs/ttrpg.crt;
    ssl_certificate_key /etc/ssl/private/ttrpg.key;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. Authentication & Authorization

**Enable Authentication:**
```bash
export ENABLE_AUTHENTICATION=true
export AUTH_SECRET_KEY=$(openssl rand -base64 32)
export AUTH_ALGORITHM=HS256
```

**Configure RBAC:**
```yaml
security:
  authorization:
    enabled: true
    rbac_enabled: true
    roles:
      - name: admin
        permissions: ["*"]
      - name: user
        permissions: ["read", "search"]
```

### 3. Data Protection

**Encryption at Rest:**
```bash
# Enable encryption
export ENABLE_ENCRYPTION=true
export ENCRYPTION_KEY=$(openssl rand -base64 32)
```

**Backup Encryption:**
```bash
# Encrypt backups
python deploy/backup/backup_manager.py --create --encrypt
```

### 4. Compliance

**Audit Logging:**
```bash
export ENABLE_AUDIT=true
export AUDIT_RETENTION_DAYS=90
export AUDIT_LOG_FILE=/var/log/ttrpg-assistant/audit.log
```

**GDPR Compliance:**
```bash
# Enable data export/deletion
export ENABLE_DATA_EXPORT=true
export ENABLE_DATA_DELETION=true
```

## Performance Tuning

### 1. Database Optimization

```bash
# Optimize ChromaDB
export CHROMA_DB_IMPL=clickhouse  # For better performance
export CONNECTION_POOL_SIZE=20
```

### 2. Caching Configuration

```bash
# Increase cache sizes
export CACHE_TTL_SECONDS=7200
export SEARCH_CACHE_SIZE=5000
export CACHE_MAX_MEMORY_MB=500
```

### 3. Parallel Processing

```bash
# Increase workers
export MAX_WORKERS=8
export EMBEDDING_BATCH_SIZE=64
```

### 4. GPU Acceleration

```bash
# Enable GPU
export ENABLE_GPU=true
export CUDA_VISIBLE_DEVICES=0,1  # Multiple GPUs
```

## Maintenance

### Regular Tasks

**Daily:**
- Check application logs
- Monitor resource usage
- Verify health endpoint

**Weekly:**
- Review security logs
- Update dependencies
- Create backups

**Monthly:**
- Clean old logs and cache
- Update language models
- Performance review

### Update Procedure

```bash
# 1. Create backup
make deploy-backup

# 2. Stop service
sudo systemctl stop ttrpg-assistant

# 3. Update code
git pull origin main

# 4. Run migrations
make deploy-migrate VERSION=1.1.0

# 5. Restart service
sudo systemctl start ttrpg-assistant

# 6. Verify
make deploy-health-check
```

## Support

For additional help:
- Documentation: https://github.com/Raudbjorn/MDMAI/wiki
- Issues: https://github.com/Raudbjorn/MDMAI/issues
- Community: Discord/Slack channels

## License

See LICENSE file for details.