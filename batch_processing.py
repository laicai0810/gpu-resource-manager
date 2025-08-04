#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Processing Examples for GPU Resource Manager
Shows how to efficiently process large datasets
"""

import numpy as np
import time
from gpu_manager import GPUManager, gpu_task, TaskPriority, SchedulingStrategy


def example_image_batch_processing():
    """Process batches of images using GPU"""
    print("\n=== Image Batch Processing Example ===")
    
    manager = GPUManager(
        min_free_memory_gb=2.0,
        scheduling_strategy=SchedulingStrategy.MEMORY_FIRST,
        max_workers=4
    )
    manager.start()
    
    # Simulate image data (batch_size, height, width, channels)
    num_images = 10000
    image_shape = (224, 224, 3)
    
    print(f"Processing {num_images} images of shape {image_shape}")
    
    def process_image_batch(batch_indices):
        """Process a batch of images"""
        import cupy as cp
        
        batch_size = len(batch_indices)
        # Simulate loading images
        images = cp.random.randn(batch_size, *image_shape, dtype=cp.float32)
        
        # Simulate image processing pipeline
        # 1. Normalization
        images = (images - cp.mean(images)) / cp.std(images)
        
        # 2. Apply filters (edge detection)
        kernel = cp.array([[-1, -1, -1], 
                          [-1,  8, -1], 
                          [-1, -1, -1]], dtype=cp.float32)
        
        # 3. Feature extraction (simplified)
        features = cp.mean(images, axis=(1, 2))  # Global average pooling
        
        return features.get()  # Transfer to CPU
    
    # Create image indices
    all_indices = list(range(num_images))
    
    # Process in batches
    start_time = time.time()
    results = manager.batch_process(
        data=all_indices,
        process_func=process_image_batch,
        batch_size=100,  # Process 100 images at a time
        required_memory_gb=2.0
    )
    
    processing_time = time.time() - start_time
    
    print(f"\n✅ Processed {len(results)} images")
    print(f"Total time: {processing_time:.2f} seconds")
    print(f"Images per second: {num_images/processing_time:.0f}")
    print(f"Feature shape: {results[0].shape}")
    
    manager.shutdown()


def example_data_transformation_pipeline():
    """Complex data transformation pipeline"""
    print("\n=== Data Transformation Pipeline Example ===")
    
    manager = GPUManager(
        min_free_memory_gb=1.0,
        scheduling_strategy=SchedulingStrategy.ADAPTIVE
    )
    manager.start()
    
    # Generate synthetic tabular data
    num_rows = 1000000
    num_features = 100
    
    print(f"Transforming dataset with {num_rows:,} rows and {num_features} features")
    
    # Pipeline stages
    @gpu_task(required_memory_gb=2.0)
    def stage1_cleaning(data_chunk):
        """Stage 1: Data cleaning and imputation"""
        import cupy as cp
        
        data = cp.asarray(data_chunk, dtype=cp.float32)
        
        # Simulate missing value imputation
        mask = cp.random.random(data.shape) < 0.1  # 10% missing
        data[mask] = cp.nan
        
        # Fill with column means
        col_means = cp.nanmean(data, axis=0)
        inds = cp.where(cp.