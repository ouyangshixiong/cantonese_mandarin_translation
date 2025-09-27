# Python Virtual Environment Configuration
# Cantonese-Mandarin Translation System

## 🐍 Python Version Requirements

**Supported Python Versions**: 3.8 - 3.11  
**Recommended Version**: 3.10.x  
**Minimum Version**: 3.8  
**Maximum Version**: 3.11 (3.12 not fully tested)

## 📋 Environment Setup

### Option 1: Standard venv (Recommended)

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Option 2: conda Environment

```bash
# Create conda environment
conda create -n cantonese_translation python=3.10

# Activate environment
conda activate cantonese_translation

# Install dependencies
pip install -r requirements.txt
```

### Option 3: pyenv (Development)

```bash
# Install Python 3.10
pyenv install 3.10.12

# Create virtual environment
pyenv virtualenv 3.10.12 cantonese_translation

# Activate environment
pyenv activate cantonese_translation

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Framework-Specific Setup

### PyTorch Environment

```bash
# For CUDA 11.8
pip install torch==2.4.1+cu118 torchvision==0.15.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# For CUDA 12.1
pip install torch==2.4.1+cu121 torchvision==0.15.0+cu121 torchaudio==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==2.4.1+cpu torchvision==0.15.0+cpu torchaudio==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### PaddlePaddle Environment

```bash
# For CUDA 11.8
python -m pip install paddlepaddle-gpu==2.6.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# For CUDA 12.0
python -m pip install paddlepaddle-gpu==2.6.0.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# CPU only
python -m pip install paddlepaddle==2.6.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

## 🏗️ Development Environment

### Development Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Or install with development extras
pip install -e ".[dev]"
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🧪 Testing Environment

### Test Dependencies

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-xdist

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run parallel tests
pytest tests/ -n auto
```

## 📊 Environment Verification

### Basic Verification

```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "(torch|paddle|transformers)"

# Verify PyTorch
cat << EOF > test_pytorch.py
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
EOF
python test_pytorch.py

# Verify PaddlePaddle
cat << EOF > test_paddle.py
import paddle
print(f"PaddlePaddle version: {paddle.__version__}")
print(f"CUDA available: {paddle.is_compiled_with_cuda()}")
if paddle.is_compiled_with_cuda():
    print(f"GPU count: {paddle.device.cuda.device_count()}")
EOF
python test_paddle.py
```

### Advanced Verification

```bash
# Run comprehensive environment check
python scripts/check_environment.py

# Verify model loading
python scripts/test_model_loading.py

# Run inference test
python scripts/inference.py --text "test" --verbose
```

## 🐳 Docker Environment

### Using Docker

```bash
# Build Docker image
docker build -t cantonese-translation:latest .

# Run with GPU
docker run --gpus all -p 8000:8000 cantonese-translation:latest

# Run CPU only
docker run -p 8000:8000 cantonese-translation:latest
```

### Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   ```bash
   # Check CUDA version
   nvidia-smi
   
   # Install compatible PyTorch
   pip install torch==2.4.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. **PaddlePaddle Installation Fails**
   ```bash
   # Update pip and setuptools
   pip install --upgrade pip setuptools
   
   # Install with specific index
   pip install paddlepaddle-gpu==2.6.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
   ```

3. **Memory Issues**
   ```bash
   # Use CPU only requirements
   pip install -r requirements-cpu.txt
   
   # Or install with memory constraints
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

4. **Import Errors**
   ```bash
   # Reinstall problematic packages
   pip uninstall torch paddlepaddle paddlenlp
   pip install torch paddlepaddle paddlenlp
   ```

### Performance Issues

1. **Slow Model Loading**
   - Pre-download models to local cache
   - Use SSD storage for model files
   - Enable model parallelism if multiple GPUs

2. **High Memory Usage**
   - Use INT8 quantization
   - Reduce batch size
   - Enable gradient checkpointing

## 📋 Environment Variables

### Recommended Settings

```bash
# PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Hugging Face cache
export HF_HOME=/path/to/hf_cache
export TRANSFORMERS_CACHE=/path/to/transformers_cache

# PaddlePaddle settings
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.8

# Logging
export LOG_LEVEL=INFO
export WANDB_PROJECT=cantonese-translation
```

## 🔒 Security Notes

- Always use virtual environments for isolation
- Regularly update dependencies for security patches
- Use specific version pins for reproducibility
- Validate downloaded models with checksums

---

**✅ 环境验证命令**: 完成上述步骤后，运行 `python scripts/check_environment.py` 验证安装成功。  
**⏱️ 预计安装时间**: 10-30分钟 (取决于网络速度和硬件配置)  
**💾 磁盘空间要求**: 约50GB (包含模型文件和依赖)  
**🐧 支持平台**: Linux, macOS, Windows (WSL2推荐)