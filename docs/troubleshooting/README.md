# TTRPG Assistant Troubleshooting Guide

This comprehensive guide helps diagnose and resolve common issues with the TTRPG Assistant MCP Server.

## Quick Diagnosis

### System Health Check

Run the diagnostic script:
```bash
python scripts/diagnose.py
```

This checks:
- Python version and dependencies
- ChromaDB connectivity
- File permissions
- Memory and disk space
- Configuration validity

## Common Issues and Solutions

## 1. Installation Issues

### Problem: Dependencies Won't Install

**Error Message:**
```
ERROR: Could not install packages due to an EnvironmentError
```

**Solutions:**

1. **Update pip:**
```bash
python -m pip install --upgrade pip
```

2. **Use virtual environment:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Install system dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# macOS
brew install python3

# Windows
# Install Visual C++ Build Tools
```

### Problem: ChromaDB Installation Fails

**Error Message:**
```
ERROR: Failed building wheel for chromadb
```

**Solutions:**

1. **Install with CPU-only requirements:**
```bash
pip install -r requirements-cpu.txt
```

2. **Manual ChromaDB installation:**
```bash
pip install chromadb==0.4.22 --no-cache-dir
```

## 2. Startup Issues

### Problem: Server Won't Start

**Error Message:**
```
ModuleNotFoundError: No module named 'fastmcp'
```

**Solutions:**

1. **Verify installation:**
```bash
pip list | grep fastmcp
# If missing:
pip install fastmcp
```

2. **Check Python path:**
```python
import sys
print(sys.path)
# Ensure project directory is included
```

3. **Verify file structure:**
```bash
ls -la src/main.py
# File should exist and be readable
```

### Problem: Database Connection Failed

**Error Message:**
```
chromadb.errors.ConnectionError: Could not connect to ChromaDB
```

**Solutions:**

1. **Check ChromaDB path:**
```python
# config/settings.py
CHROMADB_PATH = "/absolute/path/to/chromadb"
```

2. **Verify permissions:**
```bash
ls -la data/chromadb/
# Should be writable by current user
chmod 755 data/chromadb/
```

3. **Clear corrupted database:**
```bash
mv data/chromadb data/chromadb.backup
python src/main.py  # Will recreate database
```

## 3. PDF Processing Issues

### Problem: PDF Won't Process

**Error Message:**
```
PyPDF2.errors.PdfReadError: Cannot read an encrypted PDF
```

**Solutions:**

1. **Remove PDF encryption:**
```bash
# Using qpdf
qpdf --decrypt input.pdf output.pdf

# Using pdftk
pdftk input.pdf output output.pdf user_pw PASSWORD
```

2. **Try alternative parser:**
```python
# config/settings.py
PDF_PARSER = "pdfplumber"  # Instead of PyPDF2
```

### Problem: Processing Takes Too Long

**Symptoms:** PDF processing hangs or takes hours

**Solutions:**

1. **Increase timeout:**
```python
# config/settings.py
PDF_TIMEOUT_SECONDS = 600  # 10 minutes
```

2. **Process in smaller batches:**
```python
# config/settings.py
PDF_BATCH_SIZE = 10  # Pages per batch
```

3. **Enable parallel processing:**
```python
# config/settings.py
PDF_PARALLEL_WORKERS = 4
```

## 4. Search Issues

### Problem: Search Returns No Results

**Symptoms:** Valid queries return empty results

**Solutions:**

1. **Verify content was indexed:**
```python
from src.core.database import ChromaDBManager
db = ChromaDBManager()
count = db.get_collection_count("rulebooks")
print(f"Documents indexed: {count}")
```

2. **Rebuild search indices:**
```bash
python scripts/rebuild_indices.py
```

3. **Check query format:**
```python
# Use broader search terms
result = await search(query="spell", max_results=10)
```

### Problem: Search is Slow

**Symptoms:** Searches take > 5 seconds

**Solutions:**

1. **Optimize indices:**
```bash
python scripts/optimize_db.py
```

2. **Increase cache size:**
```python
# config/settings.py
CACHE_SIZE_MB = 2048
CACHE_TTL_SECONDS = 900
```

3. **Enable query caching:**
```python
# config/settings.py
ENABLE_QUERY_CACHE = True
```

## 5. Memory Issues

### Problem: High Memory Usage

**Symptoms:** Server using > 8GB RAM

**Solutions:**

1. **Limit cache size:**
```python
# config/settings.py
MAX_CACHE_ENTRIES = 1000
CACHE_SIZE_MB = 512
```

2. **Reduce batch sizes:**
```python
# config/settings.py
EMBEDDING_BATCH_SIZE = 16
SEARCH_BATCH_SIZE = 50
```

3. **Enable memory monitoring:**
```python
# config/settings.py
ENABLE_MEMORY_MONITOR = True
MEMORY_LIMIT_MB = 4096
```

### Problem: Out of Memory Errors

**Error Message:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Increase swap space:**
```bash
# Linux
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

2. **Use streaming mode:**
```python
# config/settings.py
USE_STREAMING = True
STREAM_CHUNK_SIZE = 1000
```

## 6. Performance Issues

### Problem: Slow Response Times

**Symptoms:** Tools take > 1 second to respond

**Solutions:**

1. **Profile performance:**
```bash
python -m cProfile src/main.py > profile.txt
```

2. **Enable performance monitoring:**
```python
# config/settings.py
ENABLE_PROFILING = True
LOG_SLOW_QUERIES = True
SLOW_QUERY_THRESHOLD_MS = 500
```

3. **Optimize database:**
```bash
python scripts/vacuum_db.py
python scripts/analyze_db.py
```

## 7. Campaign Management Issues

### Problem: Campaign Data Lost

**Symptoms:** Campaign disappeared or data missing

**Solutions:**

1. **Check for backups:**
```bash
ls -la backups/campaigns/
# Restore if found
python scripts/restore_campaign.py --id campaign_id
```

2. **Recover from versions:**
```python
from src.campaign.campaign_manager import CampaignManager
manager = CampaignManager()
versions = manager.get_versions("campaign_id")
manager.rollback("campaign_id", version=versions[-2])
```

### Problem: Version Conflicts

**Error Message:**
```
VersionConflictError: Campaign has been modified
```

**Solutions:**

1. **Force update:**
```python
await update_campaign_data(
    campaign_id=campaign_id,
    data_type="character",
    data=character_data,
    force=True
)
```

2. **Merge changes:**
```python
latest = await get_campaign_data(campaign_id)
# Merge your changes with latest
merged_data = merge_campaign_data(latest, your_changes)
await update_campaign_data(campaign_id, "character", merged_data)
```

## 8. Session Management Issues

### Problem: Session Won't End

**Symptoms:** Session stuck in "active" state

**Solutions:**

1. **Force end session:**
```python
from src.session.session_manager import SessionManager
manager = SessionManager()
manager.force_end_session("session_id")
```

2. **Clean up stale sessions:**
```bash
python scripts/cleanup_sessions.py --older-than 24h
```

### Problem: Initiative Order Lost

**Symptoms:** Combat order resets or disappears

**Solutions:**

1. **Restore from backup:**
```python
session = await get_session_data(session_id)
# Initiative should be in session["initiative_order"]
```

2. **Rebuild from logs:**
```bash
python scripts/rebuild_initiative.py --session-id session_id
```

## 9. Character Generation Issues

### Problem: Generation Fails

**Error Message:**
```
GenerationError: Unable to generate character
```

**Solutions:**

1. **Check system support:**
```python
supported_systems = await get_supported_systems()
print(supported_systems)
```

2. **Use default parameters:**
```python
character = await generate_character(
    system="D&D 5e",
    level=1,
    stat_method="standard"
)
```

## 10. Network Issues (Future Web UI)

### Problem: Can't Connect to Server

**Solutions:**

1. **Check firewall:**
```bash
sudo ufw status
sudo ufw allow 8000
```

2. **Verify binding:**
```python
# config/settings.py
BIND_HOST = "0.0.0.0"  # Listen on all interfaces
BIND_PORT = 8000
```

## Diagnostic Tools

### Log Analysis

```bash
# View recent errors
grep ERROR logs/server.log | tail -20

# Count error types
grep ERROR logs/server.log | cut -d: -f4 | sort | uniq -c

# Find slow queries
grep "SLOW QUERY" logs/server.log
```

### Database Diagnostics

```python
# scripts/db_diagnostics.py
from src.core.database import ChromaDBManager

db = ChromaDBManager()
print(f"Collections: {db.list_collections()}")
print(f"Total documents: {db.get_total_documents()}")
print(f"Index status: {db.check_indices()}")
```

### Performance Diagnostics

```python
# scripts/perf_diagnostics.py
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
print(f"CPU: {process.cpu_percent()}%")
print(f"Open files: {len(process.open_files())}")
print(f"Threads: {process.num_threads()}")
```

## Emergency Recovery

### Complete System Reset

```bash
#!/bin/bash
# emergency_reset.sh

# Backup everything
tar -czf emergency_backup_$(date +%Y%m%d).tar.gz data/ campaigns/ config/

# Stop server
pkill -f "python src/main.py"

# Clear caches
rm -rf cache/*
rm -rf temp/*

# Reset database
mv data/chromadb data/chromadb.old
mkdir -p data/chromadb

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Restart
python src/main.py
```

## Getting Help

### Before Asking for Help

1. **Run diagnostics:**
```bash
python scripts/diagnose.py > diagnostics.txt
```

2. **Collect logs:**
```bash
tar -czf logs.tar.gz logs/
```

3. **System information:**
```bash
python --version
pip list > pip_list.txt
uname -a > system_info.txt
```

### Where to Get Help

1. **GitHub Issues**: https://github.com/Raudbjorn/MDMAI/issues
2. **Documentation**: Check all guides in `docs/`
3. **Debug Mode**: Enable for detailed information
```python
# config/settings.py
DEBUG = True
VERBOSE_LOGGING = True
```

## Prevention Tips

### Regular Maintenance

1. **Daily**: Check logs for warnings
2. **Weekly**: Clean temporary files
3. **Monthly**: Optimize database
4. **Quarterly**: Full backup and test recovery

### Monitoring Setup

```python
# config/monitoring.py
ALERTS = {
    "high_memory": {"threshold": 0.8, "action": "email"},
    "slow_query": {"threshold": 1000, "action": "log"},
    "error_rate": {"threshold": 10, "action": "email"},
    "disk_space": {"threshold": 0.9, "action": "email"}
}
```

### Preventive Configuration

```python
# config/settings.py
# Prevent common issues
MAX_PDF_SIZE_MB = 100
MAX_CAMPAIGN_SIZE_MB = 500
AUTO_BACKUP_ENABLED = True
AUTO_CLEANUP_DAYS = 30
HEALTH_CHECK_INTERVAL = 300
```

## Next Steps

If issues persist after trying these solutions:
1. Enable debug mode
2. Collect diagnostic information
3. Create a GitHub issue with details
4. Include error messages and logs