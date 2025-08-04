#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for GPU Resource Manager
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Basic requirements
install_requires = [
    'numpy>=1.19.0',
    'psutil>=5.8.0',
    'pynvml>=11.0.0',
    'pandas>=1.3.0',
    'matplotlib>=3.3.0',
]

# Optional requirements
extras_require = {
    'cuda': ['cupy-cuda11x>=10.0.0'],  # Adjust based on CUDA version
    'web': ['gradio>=3.0.0'],
    'ml': [
        'scikit-learn>=0.24.0',
        'xgboost>=1.5.0',
        'lightgbm>=3.0.0',
    ],
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.0.0',
        'black>=21.0.0',
        'flake8>=3.9.0',
        'mypy>=0.900',
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=1.0.0',
        'sphinx-autodoc-typehints>=1.12.0',
    ],
}

# All extras
extras_require['all'] = sum(extras_require.values(), [])

setup(
    name="gpu-resource-manager",
    version="1.0.0",
    author="Tom Ricard",
    author_email="ysx_explorer@163.com",
    description="Enterprise-grade GPU resource management framework for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/laicai0810/gpu-resource-manager",
    project_urls={
        "Bug Tracker": "https://github.com/laicai0810/gpu-resource-manager/issues",
        "Documentation": "https://gpu-resource-manager.readthedocs.io",
        "Source Code": "https://github.com/laicai0810/gpu-resource-manager",
        "Author ORCID": "https://orcid.org/0009-0000-3839-9676",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware :: Symmetric Multi-processing",
        "Topic :: System :: Monitoring",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "gpu-manager=gpu_manager.cli:main",
            "gpu-monitor=gpu_manager.monitor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "gpu_manager": ["*.json", "*.yaml", "*.yml"],
    },
    keywords=[
        "gpu", "cuda", "resource-management", "scheduling", 
        "monitoring", "deep-learning", "machine-learning",
        "parallel-computing", "hpc"
    ],
    zip_safe=False,
)