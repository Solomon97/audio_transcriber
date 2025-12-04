# Installation Guide

This guide walks you through installing `audio_transcriber` and all required dependencies, including optional GPU acceleration for faster transcription.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Recommended: Using Linux or WSL](#recommended-using-linux-or-wsl)
3. [Step 1: Installing NVIDIA CUDA Libraries (GPU Support)](#step-1-installing-nvidia-cuda-libraries-gpu-support)
4. [Step 2: Installing audio_transcriber](#step-2-installing-audio_transcriber)
5. [Step 3: Installing Ollama](#step-3-installing-ollama)
6. [Verifying the Installation](#verifying-the-installation)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

- **Python**: 3.11+
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **GPU (optional)**: NVIDIA GPU with CUDA support

### Supported Operating Systems

- **Linux** (Ubuntu 20.04+, Fedora, Arch) — recommended
- **WSL** (Windows Subsystem for Linux) — recommended for Windows users
- Windows 10/11 (native, but requires more setup)
- macOS (Intel or Apple Silicon)

### Python Installation

Verify Python 3.11+ is installed:

```bash
python --version
```

If not installed: https://www.python.org/downloads/ or use your system package manager:

```bash
# Ubuntu/Debian
sudo apt install python3.11 python3.11-venv

# macOS (Homebrew)
brew install python@3.11
```

---

## Recommended: Using Linux or WSL

**We strongly recommend using Linux or WSL for this package.** Linux/WSL provides easier CUDA setup, better compatibility with faster-whisper, and simpler configuration.

### Installing WSL (Windows Users)

Official guide: https://learn.microsoft.com/en-us/windows/wsl/install

Quick install (Windows 10 version 2004+ or Windows 11):

```powershell
wsl --install
```

Restart your computer and complete the Ubuntu setup.

### WSL GPU Support

WSL 2 automatically supports CUDA if you have:
- Latest NVIDIA GPU driver installed on Windows
- No CUDA installation needed on Windows itself

> **Important**: Do NOT install NVIDIA drivers inside WSL. The Windows driver automatically provides CUDA support to WSL through a stub library. You only need to install CUDA libraries inside WSL (see Step 1).

For detailed GPU setup in WSL, see: https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute

---

## Step 1: Installing NVIDIA CUDA Libraries (GPU Support)

GPU acceleration provides ~4-10x speedup. This step is **optional** — without a GPU, `faster-whisper` runs on CPU.

### Requirements

`faster-whisper` uses CTranslate2, which requires:
- **CUDA 12.x** (12.3+ recommended)
- **cuDNN 9.x**
- **cuBLAS** (included with CUDA or pip packages)

### Option 1: Install via pip (Linux/WSL — Recommended)

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
```

**Important**: When using pip-installed CUDA libraries, you must set `LD_LIBRARY_PATH`. Add this to your `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
```

Reload your shell:

```bash
source ~/.bashrc
```

### Option 2: Install CUDA Toolkit (Linux/WSL — System-Wide)

For system-wide installation (no `LD_LIBRARY_PATH` needed):

```bash
# Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-3
sudo apt install cudnn9-cuda-12
```

Add to `~/.bashrc`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Option 3: Windows Native (Not Recommended)

If you must use Windows natively:

1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Install cuDNN via pip (modern method, no manual DLL copying):

```powershell
pip install nvidia-cudnn-cu12==9.*
```

> **Note**: Windows does not require manual PATH configuration — the CUDA installer handles this automatically.

### Option 4: Docker

```bash
docker pull nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
```

---

## Step 2: Installing audio_transcriber

### Clone the Repository

```bash
git clone <https://github.com/Solomon97/audio_transcriber>
cd audio_transcriber
```

### Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS/WSL
# venv\Scripts\activate     # Windows native
```

### Install the Package

**Basic installation** (transcription only):

```bash
pip install -e .
```

**With Ollama support** (adds Python client for LLM features):

```bash
pip install -e ".[ollama]"
```

> **Note**: This installs the `ollama` Python client library, which communicates with the Ollama server. You still need to install the Ollama application separately (Step 3).

**Development installation**:

```bash
pip install -e ".[dev,ollama]"
```

---

## Step 3: Installing Ollama

Ollama is the LLM server required for post-processing features (cleaning, summarizing, translating, extracting). Skip this step if you only need transcription.

### Linux / WSL

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify:

```bash
ollama --version
```

### macOS

```bash
brew install ollama
brew services start ollama
```

### Windows Native

Download and install from: https://ollama.com/download

### Using Windows Ollama from WSL

If Ollama is installed on Windows and you want to access it from WSL:

1. Create/edit `%USERPROFILE%\.ollama\config.toml`:

```toml
OLLAMA_HOST = "0.0.0.0"
```

2. Restart Ollama on Windows

3. In your WSL Python code:

```python
from audio_transcriber import PostProcessor
processor = PostProcessor(base_url="http://host.docker.internal:11434")
```

### Pull Recommended Models

```bash
ollama pull yi:9b          # Chinese (Traditional)
ollama pull llama3.1:8b    # English
ollama pull qwen2.5:7b     # Japanese, Korean, general
```

Verify:

```bash
ollama list
```

---

## Verifying the Installation

### Test Transcription

```python
from audio_transcriber import Transcriber

transcriber = Transcriber(model_size="large-v3")
print("Transcriber loaded successfully!")
print(f"Device: {transcriber.device}")
```

### Test GPU Support

```python
from faster_whisper import WhisperModel

try:
    model = WhisperModel("tiny", device="cuda", compute_type="float16")
    print("GPU support is working!")
except Exception as e:
    print(f"GPU not available: {e}")
```

### Test Ollama

```python
from audio_transcriber import PostProcessor

try:
    processor = PostProcessor()
    print("Ollama connected!")
except Exception as e:
    print(f"Ollama error: {e}")
```

### Run Example Script

```bash
python examples/transcribe_auto_detect.py your_audio.wav
```

---

## Troubleshooting

### LD_LIBRARY_PATH Errors (Linux/WSL)

If using pip-installed CUDA libraries and you see missing library errors:

1. Verify the export is in your `~/.bashrc`
2. Reload: `source ~/.bashrc`
3. Verify: `echo $LD_LIBRARY_PATH`

> This is only needed for pip-installed CUDA (Option 1). System-wide installs don't require this.

### DLL Not Found (Windows Native)

The CUDA installer should configure PATH automatically. If you still see errors:

1. Restart your terminal/PowerShell
2. Verify CUDA is installed: check `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\`
3. If needed, manually add to PATH via System Properties → Environment Variables

### ctranslate2 Version Mismatch

```bash
pip install --force-reinstall ctranslate2
```

For older CUDA versions:

```bash
# CUDA 11 + cuDNN 8
pip install --force-reinstall ctranslate2==3.24.0

# CUDA 12 + cuDNN 8
pip install --force-reinstall ctranslate2==4.4.0
```

### Ollama Connection Refused

```bash
ollama serve
curl http://localhost:11434/api/tags
```

### Ollama Model Not Found

```bash
ollama pull yi:9b
ollama pull qwen2.5:7b
```

### Slow CPU Transcription

Use smaller model or int8 precision:

```python
transcriber = Transcriber(model_size="medium", compute_type="int8")
```

### CUDA Out of Memory

```python
# Use smaller model
transcriber = Transcriber(model_size="medium")

# Or use int8 precision
transcriber = Transcriber(model_size="large-v3", compute_type="int8_float16")
```

---

## Quick Start Summary

```bash
# 0. (Windows users) Install WSL first
#    https://learn.microsoft.com/en-us/windows/wsl/install

# 1. Install CUDA libraries (for GPU support)
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
# Add LD_LIBRARY_PATH to ~/.bashrc (see Step 1)

# 2. Install audio_transcriber
git clone <repository-url>
cd audio_transcriber
pip install -e ".[ollama]"

# 3. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull yi:9b

# 4. Test
python -c "from audio_transcriber import Transcriber; print('Success!')"
```

---

## Getting Help

1. [faster-whisper GitHub issues](https://github.com/SYSTRAN/faster-whisper/issues)
2. [Ollama documentation](https://ollama.com/docs)
3. Contact: solomonc@alum.mit.edu