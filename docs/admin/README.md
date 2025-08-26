# TTRPG Assistant Administrator Guide

This guide covers installation, configuration, maintenance, and optimization of the TTRPG Assistant MCP Server.

## Table of Contents

1. [Installation and Setup](./installation.md)
2. [Configuration Management](./configuration.md)
3. [Database Administration](./database.md)
4. [Performance Tuning](./performance.md)
5. [Backup and Recovery](./backup.md)
6. [Security](./security.md)
7. [Monitoring](./monitoring.md)
8. [Troubleshooting](./troubleshooting.md)

## System Architecture

### Component Overview

```
┌─────────────────┐
│   MCP Client    │
└────────┬────────┘
         │ stdio
┌────────▼────────┐
│   MCP Server    │
│   (FastMCP)     │
└────────┬────────┘
         │
┌────────▼────────┐
│    ChromaDB     │
│  (Vector Store) │
└─────────────────┘
```

### Key Components

1. **MCP Server**: FastMCP-based server handling all tool requests
2. **ChromaDB**: Embedded vector database for content storage
3. **PDF Processor**: Extraction and chunking pipeline
4. **Search Engine**: Hybrid semantic and keyword search
5. **Cache Layer**: LRU cache for performance optimization

## Quick Start for Administrators

### 1. System Requirements

#### Minimum Requirements
- CPU: 4 cores
- RAM: 4GB
- Storage: 10GB free space
- Python: 3.8+
- OS: Linux, macOS, or Windows 10+

#### Recommended Requirements
- CPU: 8+ cores
- RAM: 16GB
- Storage: 50GB SSD
- Python: 3.10+
- GPU: CUDA-capable (optional)

### 2. Installation Steps

```bash
# Clone repository
git clone https://github.com/Raudbjorn/MDMAI.git
cd MDMAI

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run setup script
python scripts/setup.py
```

### 3. Initial Configuration

Edit `config/settings.py`:

```python
# Database settings
CHROMADB_PATH = "/var/lib/ttrpg/chromadb"
CHROMADB_HOST = "localhost"
CHROMADB_PORT = 8000

# Performance settings
MAX_WORKERS = 8
CACHE_SIZE_MB = 2048
BATCH_SIZE = 100

# Logging
LOG_LEVEL = "INFO"
LOG_PATH = "/var/log/ttrpg/server.log"
```

### 4. Start the Server

```bash
# Development mode
python src/main.py

# Production mode with systemd
sudo systemctl start ttrpg-assistant
sudo systemctl enable ttrpg-assistant
```

## Directory Structure

```
/opt/ttrpg-assistant/
├── config/           # Configuration files
├── data/            # ChromaDB storage
├── logs/            # Application logs
├── backups/         # Backup directory
├── cache/           # Cache storage
└── temp/            # Temporary files
```

## Environment Variables

```bash
# Required
export TTRPG_HOME=/opt/ttrpg-assistant
export CHROMADB_PATH=/var/lib/ttrpg/chromadb

# Optional
export TTRPG_LOG_LEVEL=INFO
export TTRPG_CACHE_SIZE=2048
export TTRPG_MAX_WORKERS=8
export TTRPG_GPU_ENABLED=false
```

## Service Management

### Using systemd

```ini
# /etc/systemd/system/ttrpg-assistant.service
[Unit]
Description=TTRPG Assistant MCP Server
After=network.target

[Service]
Type=simple
User=ttrpg
Group=ttrpg
WorkingDirectory=/opt/ttrpg-assistant
Environment="PATH=/opt/ttrpg-assistant/venv/bin"
ExecStart=/opt/ttrpg-assistant/venv/bin/python src/main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Service Commands

```bash
# Start/stop/restart
sudo systemctl start ttrpg-assistant
sudo systemctl stop ttrpg-assistant
sudo systemctl restart ttrpg-assistant

# Check status
sudo systemctl status ttrpg-assistant

# View logs
journalctl -u ttrpg-assistant -f
```

## Database Management

### ChromaDB Administration

```python
# Check database health
python scripts/check_db_health.py

# Optimize indices
python scripts/optimize_db.py

# Export collections
python scripts/export_collections.py --output backup.tar.gz

# Import collections
python scripts/import_collections.py --input backup.tar.gz
```

### Collection Statistics

```bash
# View collection sizes
python -c "
from src.core.database import ChromaDBManager
db = ChromaDBManager()
print(db.get_statistics())
"
```

## Performance Monitoring

### Key Metrics

1. **Response Time**: < 100ms for search queries
2. **Memory Usage**: < 4GB under normal load
3. **CPU Usage**: < 50% average
4. **Cache Hit Rate**: > 80%
5. **Database Size**: Monitor growth rate

### Monitoring Commands

```bash
# Real-time monitoring
python scripts/monitor.py --interval 5

# Generate performance report
python scripts/performance_report.py --days 7

# Check slow queries
python scripts/analyze_slow_queries.py
```

## Backup and Recovery

### Automated Backups

```bash
# Setup daily backups
crontab -e
0 2 * * * /opt/ttrpg-assistant/scripts/backup.sh
```

### Manual Backup

```bash
# Full backup
python scripts/backup.py --full --output /backups/

# Incremental backup
python scripts/backup.py --incremental --output /backups/

# Verify backup
python scripts/verify_backup.py /backups/backup_20240125.tar.gz
```

### Recovery

```bash
# Restore from backup
python scripts/restore.py --input /backups/backup_20240125.tar.gz

# Partial recovery
python scripts/restore.py --input backup.tar.gz --collections campaigns
```

## Security Hardening

### File Permissions

```bash
# Set proper ownership
sudo chown -R ttrpg:ttrpg /opt/ttrpg-assistant

# Secure configuration files
chmod 600 /opt/ttrpg-assistant/config/settings.py

# Protect database
chmod 700 /var/lib/ttrpg/chromadb
```

### Network Security

```bash
# Firewall rules (if using network mode)
sudo ufw allow from 192.168.1.0/24 to any port 8000
sudo ufw enable
```

### Access Control

```python
# config/security.py
ALLOWED_PATHS = [
    "/opt/ttrpg-assistant/data",
    "/home/*/rpg_books"
]

MAX_FILE_SIZE_MB = 100
ALLOWED_EXTENSIONS = [".pdf"]
```

## Troubleshooting Common Issues

### Issue: High Memory Usage

**Symptoms**: Server consuming excessive RAM
**Solution**:
```python
# Reduce cache size
CACHE_SIZE_MB = 512

# Lower batch size
BATCH_SIZE = 50

# Enable memory profiling
ENABLE_MEMORY_PROFILER = True
```

### Issue: Slow PDF Processing

**Symptoms**: PDFs taking too long to process
**Solution**:
```bash
# Increase workers
export TTRPG_MAX_WORKERS=16

# Enable GPU acceleration
export TTRPG_GPU_ENABLED=true
```

### Issue: Database Corruption

**Symptoms**: Errors accessing collections
**Solution**:
```bash
# Check and repair
python scripts/repair_db.py

# Rebuild indices
python scripts/rebuild_indices.py
```

## Scaling Considerations

### Horizontal Scaling

For high-load environments:

1. **Load Balancer**: HAProxy or nginx
2. **Multiple Instances**: Run multiple MCP servers
3. **Shared Storage**: NFS or distributed filesystem
4. **Cache Layer**: Redis for shared caching

### Vertical Scaling

Optimization for single server:

1. **CPU**: More cores for parallel processing
2. **RAM**: 32GB+ for large collections
3. **Storage**: NVMe SSD for fast I/O
4. **GPU**: CUDA GPU for embeddings

## Maintenance Schedule

### Daily Tasks
- Check logs for errors
- Monitor disk usage
- Verify backup completion

### Weekly Tasks
- Review performance metrics
- Clean temporary files
- Update search indices

### Monthly Tasks
- Optimize database
- Review security logs
- Test backup recovery
- Apply updates

## Update Procedures

### Safe Update Process

```bash
# 1. Backup current installation
python scripts/backup.py --full

# 2. Test in staging
cp -r /opt/ttrpg-assistant /opt/ttrpg-assistant-staging
cd /opt/ttrpg-assistant-staging
git pull
pip install -r requirements.txt
python scripts/test_all.py

# 3. Apply to production
sudo systemctl stop ttrpg-assistant
cd /opt/ttrpg-assistant
git pull
pip install -r requirements.txt
python scripts/migrate.py
sudo systemctl start ttrpg-assistant
```

## Getting Support

### Resources

- **Documentation**: This directory
- **GitHub Issues**: https://github.com/Raudbjorn/MDMAI/issues
- **Logs**: Check `/var/log/ttrpg/` for detailed logs
- **Community**: Discord/Forum (if available)

### Debug Mode

Enable detailed debugging:

```python
# config/settings.py
DEBUG = True
LOG_LEVEL = "DEBUG"
TRACE_REQUESTS = True
```

## Next Steps

- Review [Installation Guide](./installation.md) for detailed setup
- Configure using [Configuration Guide](./configuration.md)
- Optimize with [Performance Guide](./performance.md)
- Secure with [Security Guide](./security.md)