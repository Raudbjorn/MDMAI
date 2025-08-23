# GPU Support Guide

## Overview

This project uses PyTorch for embedding generation and NLP tasks. By default, we install a CPU-only version to minimize download size and disk usage. This guide explains the different installation options.

## Installation Size Comparison

| Configuration | PyTorch Size | Total Install | GPU Acceleration |
|--------------|-------------|---------------|------------------|
| CPU-only (default) | ~200MB | ~1.6GB | ❌ No |
| NVIDIA CUDA | ~2GB | ~4GB | ✅ Yes (NVIDIA) |
| AMD ROCm | ~1.5GB | ~3.5GB | ✅ Yes (AMD) |

## Automatic Detection

The `quick_setup.sh` script automatically detects your hardware:
- Checks for `nvidia-smi` (NVIDIA GPUs)
- Checks for `rocm-smi` (AMD GPUs with ROCm)
- Falls back to CPU-only if no GPU is detected

## Manual Installation Options

### CPU-Only (Recommended for most users)
```bash
make install-dev  # Default, includes CPU-only PyTorch
# or explicitly:
make install-cpu-torch
```

**Pros:**
- Small download size (~200MB vs ~2GB)
- No CUDA/GPU driver requirements
- Works on all systems
- Sufficient for most TTRPG Assistant tasks

**Cons:**
- Slower embedding generation for large documents
- No GPU acceleration

### NVIDIA GPU Support
```bash
make install-cuda
```

**Requirements:**
- NVIDIA GPU (GTX 1050 or newer recommended)
- NVIDIA drivers installed
- CUDA toolkit (optional, included in PyTorch)

**Pros:**
- 5-10x faster embedding generation
- Better for processing large rulebooks
- Parallel processing capabilities

**Cons:**
- Large download (~2GB)
- Requires NVIDIA GPU
- Higher memory usage

### AMD GPU Support
```bash
make install-rocm
```

**Requirements:**
- AMD GPU (RX 5000 series or newer)
- ROCm installed (Linux only)
- Kernel support for ROCm

**Pros:**
- GPU acceleration on AMD hardware
- Good performance for embeddings

**Cons:**
- Limited OS support (Linux only)
- Requires ROCm installation
- Less mature than CUDA

## Performance Impact

For TTRPG Assistant, GPU acceleration primarily affects:

1. **PDF Processing**: Initial embedding generation for rulebooks
   - CPU: ~10-30 seconds per 100 pages
   - GPU: ~2-5 seconds per 100 pages

2. **Search Operations**: Query embedding generation
   - CPU: ~100-200ms per query
   - GPU: ~20-50ms per query

3. **Batch Operations**: Processing multiple documents
   - CPU: Sequential processing
   - GPU: Parallel processing capability

## Recommendations

### Use CPU-only if:
- You have limited disk space
- You don't have a compatible GPU
- You're primarily using pre-indexed content
- You're on a laptop or low-power system
- You want faster installation

### Use GPU support if:
- You frequently process new PDFs
- You have many large rulebooks (500+ pages)
- You need real-time response for searches
- You have a compatible GPU available
- You don't mind larger downloads

## Switching Between Versions

You can switch between CPU and GPU versions at any time:

```bash
# Switch to CPU-only
make install-cpu-torch

# Switch to NVIDIA GPU
make install-cuda

# Switch to AMD GPU
make install-rocm
```

**Note:** Switching will download and reinstall PyTorch, which may take a few minutes.

## Troubleshooting

### "CUDA out of memory" errors
- Reduce batch size in settings
- Use CPU-only version
- Close other GPU applications

### ROCm not detected
- Ensure ROCm is installed: `rocm-smi`
- Check kernel compatibility
- Use CPU-only as fallback

### Slow performance with CPU
- Normal for large documents
- Consider preprocessing during off-hours
- Use cached results when possible

## Environment Variables

You can force a specific installation:

```bash
# Force CPU-only even with GPU present
FORCE_CPU=1 ./quick_setup.sh

# Skip GPU detection
NO_GPU_DETECT=1 make install-dev
```

## Docker Considerations

The Docker image uses CPU-only by default for portability. To use GPU in Docker:

```dockerfile
# For NVIDIA GPUs, use nvidia-docker runtime
docker run --gpus all ttrpg-assistant

# Build with CUDA support
docker build --build-arg GPU_SUPPORT=cuda -t ttrpg-assistant .
```