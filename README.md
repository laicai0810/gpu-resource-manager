# GPU Resource Manager

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0000--3839--9676-green.svg)](https://orcid.org/0009-0000-3839-9676)

An enterprise-grade GPU resource management framework for Python, providing intelligent scheduling, real-time monitoring, and automatic memory management for CUDA-enabled applications.

**Author**: Tom Ricard ([@laicai0810](https://github.com/laicai0810))

## 🌟 Features

- **🚀 Intelligent Task Scheduling**: Multiple scheduling strategies including adaptive, round-robin, and performance-based
- **📊 Real-time Monitoring**: Track GPU utilization, memory usage, temperature, and power consumption
- **🔄 Automatic Resource Management**: Smart memory allocation, cleanup, and process management
- **🌐 Web Dashboard**: Interactive Gradio-based UI for monitoring and control
- **⚡ High Performance**: Multi-threaded execution with process pooling support
- **🛡️ Enterprise Ready**: Comprehensive error handling, logging, and graceful shutdown
- **🔧 Easy Integration**: Simple decorator-based API and batch processing support

## 📋 Requirements

- Python 3.8+
- CUDA 11.0+
- NVIDIA GPU with CUDA support

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/laicai0810/gpu-resource-manager.git
cd gpu-resource-manager

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from gpu_manager import GPUManager, gpu_task, TaskPriority

# Initialize the GPU manager
manager = GPUManager(
    min_free_memory_gb=2.0,
    scheduling_strategy="adaptive",
    max_workers=4
)
manager.start()

# Method 1: Using decorator
@gpu_task(required_memory_gb=4.0, priority=TaskPriority.HIGH)
def train_model(data, epochs):
    # Your GPU-intensive code here
    import cupy as cp
    # ... training logic ...
    return model

# Method 2: Direct task submission
task_id = manager.submit_task(
    func=your_gpu_function,
    args=(arg1, arg2),
    kwargs={'param': value},
    required_memory_gb=4.0,
    priority=TaskPriority.NORMAL
)
result = manager.get_task_result(task_id)

# Method 3: Batch processing
results = manager.batch_process(
    data=large_dataset,
    process_func=process_batch,
    batch_size=None,  # Auto-calculate optimal batch size
    required_memory_gb=2.0
)
```

### Web Dashboard

```python
from gpu_manager import GPUManager, GPUWebUI

# Start the web interface
manager = GPUManager()
manager.start()

web_ui = GPUWebUI(manager)
interface = web_ui.create_interface()
interface.launch(server_name="0.0.0.0", server_port=7860, share=True)
```

## 📊 Monitoring & Visualization

The web dashboard provides:
- Real-time GPU metrics visualization
- Task queue management
- Process monitoring and control
- Configuration management
- Performance history charts

![Dashboard Screenshot](docs/images/dashboard.png)

## 🏗️ Architecture

```
gpu-resource-manager/
├── gpu_manager.py          # Core GPU management framework
├── examples/
│   ├── basic_usage.py      # Basic usage examples
│   ├── batch_processing.py # Batch processing examples
│   ├── ml_training.py      # Machine learning integration
│   └── xgboost_tuning.py   # XGBoost hyperparameter tuning
├── tests/
│   ├── test_scheduler.py   # Scheduler unit tests
│   ├── test_memory.py      # Memory management tests
│   └── test_integration.py # Integration tests
├── docs/
│   ├── API.md             # API documentation
│   ├── GUIDE.md           # User guide
│   └── ADVANCED.md        # Advanced usage
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
└── README.md              # This file
```

## 🔧 Advanced Features

### Custom Scheduling Strategies

```python
from gpu_manager import SchedulingStrategy

# Use different scheduling strategies
manager = GPUManager(scheduling_strategy=SchedulingStrategy.MEMORY_FIRST)
# Options: ROUND_ROBIN, LEAST_LOADED, MEMORY_FIRST, PERFORMANCE_FIRST, ADAPTIVE
```

### Resource Limits & Priorities

```python
# Set task priorities
@gpu_task(
    required_memory_gb=8.0,
    priority=TaskPriority.CRITICAL,
    timeout=300,  # 5 minutes timeout
    max_retries=3
)
def critical_computation():
    pass
```

### Process-Level Monitoring

```python
# Get current process GPU usage
status = manager.get_status()
print(f"Current process GPU usage: {status['current_process_usage']}")

# Monitor all GPU processes
for gpu in status['gpus']:
    print(f"GPU {gpu['index']}: {len(gpu['processes'])} processes")
```

## 📈 Performance Optimization

1. **Automatic Batching**: The framework automatically determines optimal batch sizes based on available GPU memory
2. **Memory Pool Management**: Efficient memory allocation and reuse
3. **Smart Cleanup**: Automatic garbage collection and memory defragmentation
4. **Load Balancing**: Intelligent distribution of tasks across available GPUs

## 🛠️ Configuration

### Environment Variables

```bash
export GPU_MANAGER_MIN_MEMORY_GB=2.0
export GPU_MANAGER_LOG_LEVEL=INFO
export GPU_MANAGER_MAX_WORKERS=4
```

### Configuration File

```python
config = {
    "min_free_memory_gb": 2.0,
    "scheduling_strategy": "adaptive",
    "max_workers": 4,
    "enable_monitoring": True,
    "monitoring_interval": 5,  # seconds
    "task_timeout": 3600,      # 1 hour default
    "max_retries": 3
}

manager = GPUManager(**config)
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test suite
python -m pytest tests/test_scheduler.py

# Run with coverage
python -m pytest --cov=gpu_manager tests/
```

## 📚 Examples

Check the `examples/` directory for:
- Basic usage patterns
- Integration with ML frameworks (TensorFlow, PyTorch, XGBoost)
- Batch processing scenarios
- Multi-GPU training examples
- Real-world use cases

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA for CUDA and pynvml
- The CuPy team for GPU-accelerated computing
- The Gradio team for the amazing UI framework

## 📞 Support

- **Documentation**: [Full documentation](https://gpu-resource-manager.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/laicai0810/gpu-resource-manager/issues)
- **Discussions**: [GitHub Discussions](https://github.com/laicai0810/gpu-resource-manager/discussions)
- **Email**: ysx_explorer@163.com

## 🚦 Status

- Build: ![Build Status](https://img.shields.io/github/workflow/status/laicai0810/gpu-resource-manager/CI)
- Coverage: ![Coverage](https://img.shields.io/codecov/c/github/laicai0810/gpu-resource-manager)
- Downloads: ![Downloads](https://img.shields.io/pypi/dm/gpu-resource-manager)

---

**Repository name**: `gpu-resource-manager`

**Description**: Enterprise-grade GPU resource management framework for Python with intelligent scheduling, monitoring, and automatic memory management

**Author**: Tom Ricard | [ORCID](https://orcid.org/0009-0000-3839-9676) | [GitHub](https://github.com/laicai0810)
