#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Usage Examples for GPU Resource Manager
"""

import numpy as np
import time
from gpu_manager import (
    GPUManager, 
    gpu_task, 
    TaskPriority,
    SchedulingStrategy,
    init_gpu_manager,
    get_gpu_manager
)

def example_1_simple_task():
    """Example 1: Simple GPU task submission"""
    print("\n=== Example 1: Simple Task Submission ===")
    
    # Initialize GPU manager
    manager = GPUManager(min_free_memory_gb=1.0)
    manager.start()
    
    # Define a simple GPU task
    def matrix_multiplication(size=1000):
        import cupy as cp
        a = cp.random.randn(size, size, dtype=cp.float32)
        b = cp.random.randn(size, size, dtype=cp.float32)
        result = cp.dot(a, b)
        return float(cp.sum(result))
    
    # Submit task
    task_id = manager.submit_task(
        func=matrix_multiplication,
        args=(2000,),
        required_memory_gb=2.0,
        priority=TaskPriority.NORMAL
    )
    
    print(f"Task submitted with ID: {task_id}")
    
    # Get result
    result = manager.get_task_result(task_id)
    print(f"Task result: {result}")
    
    # Print status
    manager.print_status()
    
    # Shutdown
    manager.shutdown()

def example_2_decorator_usage():
    """Example 2: Using the @gpu_task decorator"""
    print("\n=== Example 2: Decorator Usage ===")
    
    # Initialize default manager
    init_gpu_manager(min_free_memory_gb=1.0)
    
    # Define GPU functions with decorator
    @gpu_task(required_memory_gb=1.0, priority=TaskPriority.HIGH)
    def compute_eigenvalues(size=500):
        import cupy as cp
        matrix = cp.random.randn(size, size, dtype=cp.float32)
        matrix = (matrix + matrix.T) / 2  # Make symmetric
        eigenvalues = cp.linalg.eigvalsh(matrix)
        return eigenvalues.get()  # Transfer to CPU
    
    @gpu_task(required_memory_gb=0.5, priority=TaskPriority.LOW)
    def vector_norm(size=1000000):
        import cupy as cp
        vector = cp.random.randn(size, dtype=cp.float32)
        norm = cp.linalg.norm(vector)
        return float(norm)
    
    # Execute tasks
    print("Computing eigenvalues...")
    eigenvalues = compute_eigenvalues(1000)
    print(f"Eigenvalues shape: {eigenvalues.shape}")
    print(f"Min eigenvalue: {np.min(eigenvalues):.4f}")
    print(f"Max eigenvalue: {np.max(eigenvalues):.4f}")
    
    print("\nComputing vector norm...")
    norm = vector_norm(10000000)
    print(f"Vector norm: {norm:.4f}")
    
    # Get manager status
    manager = get_gpu_manager()
    manager.print_status()
    manager.shutdown()

def example_3_batch_processing():
    """Example 3: Batch processing large datasets"""
    print("\n=== Example 3: Batch Processing ===")
    
    manager = GPUManager(
        min_free_memory_gb=1.0,
        scheduling_strategy=SchedulingStrategy.MEMORY_FIRST
    )
    manager.start()
    
    # Generate large dataset
    data_size = 1000
    data = [np.random.randn(100, 100) for _ in range(data_size)]
    print(f"Generated {len(data)} matrices")
    
    # Define processing function
    def process_batch(batch_data):
        import cupy as cp
        results = []
        for matrix in batch_data:
            # Transfer to GPU
            gpu_matrix = cp.asarray(matrix, dtype=cp.float32)
            # Compute SVD
            u, s, v = cp.linalg.svd(gpu_matrix)
            # Get singular values
            results.append(s.get())
        return results
    
    # Batch process
    print("Starting batch processing...")
    start_time = time.time()
    
    results = manager.batch_process(
        data=data,
        process_func=process_batch,
        batch_size=None,  # Auto-calculate optimal batch size
        required_memory_gb=2.0,
        priority=TaskPriority.NORMAL
    )
    
    elapsed_time = time.time() - start_time
    print(f"Processed {len(results)} items in {elapsed_time:.2f} seconds")
    print(f"Average time per item: {elapsed_time/len(results)*1000:.2f} ms")
    
    manager.shutdown()

def example_4_multi_gpu_scheduling():
    """Example 4: Multi-GPU task scheduling"""
    print("\n=== Example 4: Multi-GPU Scheduling ===")
    
    # Use adaptive scheduling for multi-GPU
    manager = GPUManager(
        min_free_memory_gb=1.0,
        scheduling_strategy=SchedulingStrategy.ADAPTIVE,
        max_workers=8  # More workers for parallel execution
    )
    manager.start()
    
    # Define compute-intensive task
    def train_mini_model(data_size, iterations):
        import cupy as cp
        
        # Simulate model training
        data = cp.random.randn(data_size, 100, dtype=cp.float32)
        weights = cp.random.randn(100, 10, dtype=cp.float32)
        
        for i in range(iterations):
            # Forward pass
            output = cp.dot(data, weights)
            # Simulate loss computation
            loss = cp.mean(cp.square(output))
            # Simulate backward pass
            grad = 2 * cp.dot(data.T, output) / data_size
            weights -= 0.01 * grad
            
        return float(loss)
    
    # Submit multiple tasks
    print("Submitting multiple training tasks...")
    task_ids = []
    
    for i in range(10):
        task_id = manager.submit_task(
            func=train_mini_model,
            args=(1000 + i * 100, 50),
            required_memory_gb=1.0,
            priority=TaskPriority.NORMAL if i % 2 == 0 else TaskPriority.HIGH
        )
        task_ids.append(task_id)
        print(f"Submitted task {i+1}/10")
    
    # Wait for results
    print("\nWaiting for results...")
    results = []
    for i, task_id in enumerate(task_ids):
        result = manager.get_task_result(task_id)
        results.append(result)
        print(f"Task {i+1} completed with loss: {result:.6f}")
    
    # Show final status
    manager.print_status()
    
    # Print metrics
    status = manager.get_status()
    print(f"\nTotal tasks completed: {status['completed_tasks']}")
    print(f"Failed tasks: {status['failed_tasks']}")
    
    manager.shutdown()

def example_5_error_handling():
    """Example 5: Error handling and retries"""
    print("\n=== Example 5: Error Handling ===")
    
    manager = GPUManager(min_free_memory_gb=0.5)
    manager.start()
    
    # Define a task that might fail
    fail_count = 0
    def unreliable_task(threshold=0.5):
        import cupy as cp
        nonlocal fail_count
        
        # Simulate random failures
        if np.random.random() < threshold and fail_count < 2:
            fail_count += 1
            raise RuntimeError(f"Random failure (attempt {fail_count})")
        
        # Normal computation
        data = cp.random.randn(1000, 1000)
        result = cp.sum(data)
        return float(result)
    
    # Define callbacks
    def success_callback(result):
        print(f"✅ Task succeeded with result: {result}")
    
    def error_callback(error):
        print(f"❌ Task failed with error: {error}")
    
    # Submit task with retries
    task_id = manager.submit_task(
        func=unreliable_task,
        args=(0.7,),  # 70% chance of failure
        required_memory_gb=0.5,
        max_retries=3,
        timeout=10.0,
        callback=success_callback,
        error_callback=error_callback
    )
    
    # Get result (will retry automatically)
    try:
        result = manager.get_task_result(task_id, timeout=30)
        print(f"Final result: {result}")
    except Exception as e:
        print(f"Task ultimately failed: {e}")
    
    manager.shutdown()

def example_6_resource_monitoring():
    """Example 6: Real-time resource monitoring"""
    print("\n=== Example 6: Resource Monitoring ===")
    
    manager = GPUManager(
        min_free_memory_gb=1.0,
        enable_monitoring=True
    )
    manager.start()
    
    # Submit a long-running task
    def memory_stress_test(duration=10):
        import cupy as cp
        
        allocations = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Allocate memory
            size = np.random.randint(100, 500)
            data = cp.random.randn(size, size, dtype=cp.float32)
            allocations.append(data)
            
            # Perform computation
            result = cp.dot(data, data.T)
            
            # Clean up old allocations
            if len(allocations) > 5:
                allocations.pop(0)
            
            time.sleep(0.5)
        
        return len(allocations)
    
    # Submit stress test
    print("Starting memory stress test...")
    task_id = manager.submit_task(
        func=memory_stress_test,
        args=(15,),  # 15 seconds
        required_memory_gb=2.0
    )
    
    # Monitor while running
    print("\nMonitoring GPU status during execution:")
    for i in range(5):
        time.sleep(3)
        status = manager.get_status()
        
        print(f"\n--- Status Update {i+1} ---")
        for gpu in status['gpus']:
            print(f"GPU {gpu['index']}: "
                  f"Util={gpu['utilization']:.0f}%, "
                  f"Mem={gpu['memory_used']/1024:.1f}/{gpu['memory_total']/1024:.1f}GB, "
                  f"Temp={gpu['temperature']:.0f}°C, "
                  f"Power={gpu['power_usage']:.0f}W")
    
    # Get result
    result = manager.get_task_result(task_id)
    print(f"\nStress test completed. Final allocations: {result}")
    
    manager.shutdown()

if __name__ == "__main__":
    # Run all examples
    examples = [
        example_1_simple_task,
        example_2_decorator_usage,
        example_3_batch_processing,
        example_4_multi_gpu_scheduling,
        example_5_error_handling,
        example_6_resource_monitoring
    ]
    
    for example in examples:
        try:
            example()
            time.sleep(2)  # Pause between examples
        except Exception as e:
            print(f"Example failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ All examples completed!")