#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enterprise GPU Resource Management Framework
企业级GPU资源管理框架 v3.0

Features:
- 完整的API接口设计
- 智能任务调度与负载均衡
- 实时监控与可视化
- 自动内存管理与优化
- 分布式任务处理
- 进程级资源追踪
"""

import subprocess
import os
import sys
import json
import time
import psutil
import threading
import queue
import logging
import traceback
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from functools import wraps, lru_cache
import pickle
import gc
import weakref
import signal
import atexit

try:
    import cupy as cp
    import pynvml
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠️ CUDA/CuPy not available. Running in CPU-only mode.")

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("⚠️ Gradio not available. Web UI disabled.")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('gpu_manager.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """任务优先级枚举"""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    IDLE = 1


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SchedulingStrategy(Enum):
    """调度策略枚举"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    MEMORY_FIRST = "memory_first"
    PERFORMANCE_FIRST = "performance_first"
    ADAPTIVE = "adaptive"


@dataclass
class GPUInfo:
    """GPU信息数据类"""
    index: int
    name: str
    uuid: str
    memory_total: int  # MB
    memory_free: int   # MB
    memory_used: int   # MB
    utilization: float  # %
    temperature: float  # °C
    power_usage: float  # W
    processes: List[Dict] = field(default_factory=list)
    compute_capability: Tuple[int, int] = (0, 0)
    
    @property
    def memory_usage_percent(self) -> float:
        return (self.memory_used / self.memory_total * 100) if self.memory_total > 0 else 0
    
    @property
    def available_memory_gb(self) -> float:
        return self.memory_free / 1024.0


@dataclass
class GPUTask:
    """GPU任务数据类"""
    task_id: str
    func: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    required_memory_gb: float = 1.0
    max_retries: int = 3
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    
    # 运行时属性
    status: TaskStatus = TaskStatus.PENDING
    assigned_gpu: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    
    @property
    def execution_time(self) -> Optional[timedelta]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class GPUProcessMonitor:
    """GPU进程监控器"""
    
    def __init__(self):
        self.process_cache = {}
        self._lock = threading.Lock()
        
    def get_gpu_processes(self, gpu_index: int) -> List[Dict]:
        """获取指定GPU上的进程信息"""
        processes = []
        if not CUDA_AVAILABLE:
            return processes
            
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            
            for proc in procs:
                pid = proc.pid
                used_memory = proc.usedGpuMemory // (1024 * 1024)  # MB
                
                # 获取进程详细信息
                try:
                    p = psutil.Process(pid)
                    with self._lock:
                        self.process_cache[pid] = {
                            'pid': pid,
                            'name': p.name(),
                            'username': p.username(),
                            'create_time': p.create_time(),
                            'memory_mb': used_memory,
                            'cpu_percent': p.cpu_percent(),
                            'memory_percent': p.memory_percent(),
                            'status': p.status(),
                            'cmdline': ' '.join(p.cmdline()[:50])  # 限制命令行长度
                        }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
                if pid in self.process_cache:
                    processes.append(self.process_cache[pid])
                    
        except Exception as e:
            logger.error(f"Failed to get GPU processes: {e}")
            
        return processes
    
    def get_current_process_gpu_usage(self) -> Dict[int, int]:
        """获取当前进程的GPU使用情况"""
        current_pid = os.getpid()
        gpu_usage = {}
        
        if not CUDA_AVAILABLE:
            return gpu_usage
            
        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                
                for proc in procs:
                    if proc.pid == current_pid:
                        gpu_usage[i] = proc.usedGpuMemory // (1024 * 1024)
                        
        except Exception as e:
            logger.error(f"Failed to get current process GPU usage: {e}")
            
        return gpu_usage


class IntelligentScheduler:
    """智能任务调度器"""
    
    def __init__(self, strategy: SchedulingStrategy = SchedulingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.task_history = deque(maxlen=1000)
        self.gpu_load_history = defaultdict(lambda: deque(maxlen=100))
        self._lock = threading.Lock()
        
    def select_gpu(self, task: GPUTask, available_gpus: List[GPUInfo]) -> Optional[int]:
        """根据策略选择最佳GPU"""
        if not available_gpus:
            return None
            
        with self._lock:
            if self.strategy == SchedulingStrategy.ROUND_ROBIN:
                return self._round_robin_select(available_gpus)
            elif self.strategy == SchedulingStrategy.LEAST_LOADED:
                return self._least_loaded_select(available_gpus)
            elif self.strategy == SchedulingStrategy.MEMORY_FIRST:
                return self._memory_first_select(task, available_gpus)
            elif self.strategy == SchedulingStrategy.PERFORMANCE_FIRST:
                return self._performance_first_select(available_gpus)
            elif self.strategy == SchedulingStrategy.ADAPTIVE:
                return self._adaptive_select(task, available_gpus)
                
        return available_gpus[0].index
    
    def _round_robin_select(self, gpus: List[GPUInfo]) -> int:
        """轮询选择"""
        # 简单实现：选择负载最低的GPU
        return min(gpus, key=lambda g: len(g.processes)).index
    
    def _least_loaded_select(self, gpus: List[GPUInfo]) -> int:
        """选择负载最低的GPU"""
        return min(gpus, key=lambda g: g.utilization).index
    
    def _memory_first_select(self, task: GPUTask, gpus: List[GPUInfo]) -> int:
        """优先选择内存充足的GPU"""
        suitable_gpus = [g for g in gpus if g.available_memory_gb >= task.required_memory_gb]
        if suitable_gpus:
            return max(suitable_gpus, key=lambda g: g.memory_free).index
        return gpus[0].index
    
    def _performance_first_select(self, gpus: List[GPUInfo]) -> int:
        """优先选择性能最好的GPU"""
        # 综合考虑利用率、温度和功耗
        def gpu_score(gpu: GPUInfo) -> float:
            util_score = 100 - gpu.utilization
            temp_score = max(0, 85 - gpu.temperature)  # 85°C为警戒温度
            power_score = 100 - (gpu.power_usage / 300 * 100)  # 假设300W为最大功耗
            return util_score * 0.5 + temp_score * 0.3 + power_score * 0.2
            
        return max(gpus, key=gpu_score).index
    
    def _adaptive_select(self, task: GPUTask, gpus: List[GPUInfo]) -> int:
        """自适应选择策略"""
        # 根据任务特征和历史数据智能选择
        if task.required_memory_gb > 8:  # 大内存任务
            return self._memory_first_select(task, gpus)
        elif task.priority == TaskPriority.CRITICAL:  # 关键任务
            return self._performance_first_select(gpus)
        else:  # 普通任务
            return self._least_loaded_select(gpus)
    
    def update_history(self, task: GPUTask, gpu_info: GPUInfo):
        """更新调度历史"""
        with self._lock:
            self.task_history.append({
                'task_id': task.task_id,
                'gpu_index': gpu_info.index,
                'timestamp': datetime.now(),
                'execution_time': task.execution_time,
                'memory_used': task.required_memory_gb,
                'status': task.status
            })
            
            self.gpu_load_history[gpu_info.index].append({
                'timestamp': datetime.now(),
                'utilization': gpu_info.utilization,
                'memory_used': gpu_info.memory_used,
                'temperature': gpu_info.temperature
            })


class GPUResourcePool:
    """GPU资源池管理器"""
    
    def __init__(self, min_free_memory_gb: float = 2.0):
        self.min_free_memory_gb = min_free_memory_gb
        self.gpu_locks = {}
        self.allocation_map = {}
        self._lock = threading.Lock()
        self.process_monitor = GPUProcessMonitor()
        
        if CUDA_AVAILABLE:
            pynvml.nvmlInit()
            self._init_gpu_locks()
            
    def _init_gpu_locks(self):
        """初始化GPU锁"""
        gpu_count = pynvml.nvmlDeviceGetCount() if CUDA_AVAILABLE else 0
        for i in range(gpu_count):
            self.gpu_locks[i] = threading.Semaphore(1)
            
    def get_all_gpu_info(self) -> List[GPUInfo]:
        """获取所有GPU的详细信息"""
        gpus = []
        if not CUDA_AVAILABLE:
            return gpus
            
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # 基础信息
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                uuid = pynvml.nvmlDeviceGetUUID(handle).decode('utf-8')
                
                # 内存信息
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total = mem_info.total // (1024 * 1024)
                memory_free = mem_info.free // (1024 * 1024)
                memory_used = mem_info.used // (1024 * 1024)
                
                # 利用率
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                
                # 温度
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = 0.0
                
                # 功耗
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                except:
                    power_usage = 0.0
                
                # 计算能力
                try:
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_capability = (major, minor)
                except:
                    compute_capability = (0, 0)
                
                # 进程信息
                processes = self.process_monitor.get_gpu_processes(i)
                
                gpu_info = GPUInfo(
                    index=i,
                    name=name,
                    uuid=uuid,
                    memory_total=memory_total,
                    memory_free=memory_free,
                    memory_used=memory_used,
                    utilization=utilization,
                    temperature=temperature,
                    power_usage=power_usage,
                    processes=processes,
                    compute_capability=compute_capability
                )
                
                gpus.append(gpu_info)
                
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            
        return gpus
    
    def allocate_gpu(self, required_memory_gb: float, 
                      timeout: Optional[float] = None) -> Optional[int]:
        """分配GPU资源"""
        start_time = time.time()
        
        while True:
            with self._lock:
                gpus = self.get_all_gpu_info()
                available_gpus = [
                    gpu for gpu in gpus 
                    if gpu.available_memory_gb >= max(required_memory_gb, self.min_free_memory_gb)
                    and gpu.index not in self.allocation_map
                ]
                
                if available_gpus:
                    # 选择最佳GPU
                    best_gpu = max(available_gpus, key=lambda g: g.memory_free)
                    gpu_index = best_gpu.index
                    
                    if self.gpu_locks[gpu_index].acquire(blocking=False):
                        self.allocation_map[gpu_index] = {
                            'allocated_at': datetime.now(),
                            'required_memory_gb': required_memory_gb,
                            'pid': os.getpid()
                        }
                        return gpu_index
                        
            # 检查超时
            if timeout and (time.time() - start_time) > timeout:
                return None
                
            time.sleep(0.1)
    
    def release_gpu(self, gpu_index: int):
        """释放GPU资源"""
        with self._lock:
            if gpu_index in self.allocation_map:
                del self.allocation_map[gpu_index]
                self.gpu_locks[gpu_index].release()
                
                # 清理GPU内存
                self.cleanup_gpu_memory(gpu_index)
    
    def cleanup_gpu_memory(self, gpu_index: int):
        """清理GPU内存"""
        if not CUDA_AVAILABLE:
            return
            
        try:
            with cp.cuda.Device(gpu_index):
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as e:
            logger.error(f"Failed to cleanup GPU {gpu_index} memory: {e}")


class TaskExecutor:
    """任务执行器"""
    
    def __init__(self, resource_pool: GPUResourcePool):
        self.resource_pool = resource_pool
        self.current_tasks = {}
        self._lock = threading.Lock()
        
    def execute_task(self, task: GPUTask, gpu_index: int) -> Any:
        """在指定GPU上执行任务"""
        task.assigned_gpu = gpu_index
        task.start_time = datetime.now()
        task.status = TaskStatus.RUNNING
        
        try:
            with self._lock:
                self.current_tasks[task.task_id] = task
                
            # 设置GPU设备
            if CUDA_AVAILABLE:
                cp.cuda.Device(gpu_index).use()
                
            # 执行任务
            if task.timeout:
                # 使用超时控制
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(task.func, *task.args, **task.kwargs)
                    result = future.result(timeout=task.timeout)
            else:
                result = task.func(*task.args, **task.kwargs)
                
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now()
            
            # 执行回调
            if task.callback:
                task.callback(result)
                
            return result
            
        except Exception as e:
            task.error = e
            task.status = TaskStatus.FAILED
            task.end_time = datetime.now()
            
            # 执行错误回调
            if task.error_callback:
                task.error_callback(e)
                
            raise
            
        finally:
            with self._lock:
                if task.task_id in self.current_tasks:
                    del self.current_tasks[task.task_id]
                    
            # 释放GPU资源
            self.resource_pool.release_gpu(gpu_index)


class GPUManager:
    """GPU管理器主类 - 提供完整的API接口"""
    
    def __init__(self, 
                 min_free_memory_gb: float = 2.0,
                 scheduling_strategy: SchedulingStrategy = SchedulingStrategy.ADAPTIVE,
                 max_workers: int = 4,
                 enable_monitoring: bool = True):
        """
        初始化GPU管理器
        
        Args:
            min_free_memory_gb: 最小可用内存要求
            scheduling_strategy: 调度策略
            max_workers: 最大工作线程数
            enable_monitoring: 是否启用监控
        """
        self.resource_pool = GPUResourcePool(min_free_memory_gb)
        self.scheduler = IntelligentScheduler(scheduling_strategy)
        self.executor = TaskExecutor(self.resource_pool)
        self.task_queue = queue.PriorityQueue()
        self.max_workers = max_workers
        self.enable_monitoring = enable_monitoring
        
        self.running = False
        self.workers = []
        self.monitor_thread = None
        self.task_results = {}
        self.metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_execution_time': timedelta(0),
            'gpu_utilization_history': defaultdict(list)
        }
        
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        
        # 注册清理函数
        atexit.register(self.shutdown)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("GPU Manager initialized successfully")
        
    def start(self):
        """启动GPU管理器"""
        if self.running:
            return
            
        self.running = True
        
        # 启动工作线程
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, 
                                    name=f"GPUWorker-{i}",
                                    daemon=True)
            worker.start()
            self.workers.append(worker)
            
        # 启动监控线程
        if self.enable_monitoring:
            self.monitor_thread = threading.Thread(target=self._monitor_loop,
                                                 name="GPUMonitor",
                                                 daemon=True)
            self.monitor_thread.start()
            
        logger.info(f"GPU Manager started with {self.max_workers} workers")
        
    def _worker_loop(self):
        """工作线程主循环"""
        while self.running:
            try:
                # 获取任务（优先级，任务）
                _, task = self.task_queue.get(timeout=1)
                
                # 重试逻辑
                while task.retry_count <= task.max_retries:
                    try:
                        # 获取可用GPU
                        gpus = self.resource_pool.get_all_gpu_info()
                        available_gpus = [
                            gpu for gpu in gpus 
                            if gpu.available_memory_gb >= task.required_memory_gb
                        ]
                        
                        if not available_gpus:
                            logger.warning(f"No available GPU for task {task.task_id}")
                            time.sleep(1)
                            continue
                            
                        # 选择GPU
                        gpu_index = self.scheduler.select_gpu(task, available_gpus)
                        if gpu_index is None:
                            continue
                            
                        # 分配GPU
                        allocated_gpu = self.resource_pool.allocate_gpu(
                            task.required_memory_gb, 
                            timeout=30
                        )
                        
                        if allocated_gpu is not None:
                            # 执行任务
                            result = self.executor.execute_task(task, allocated_gpu)
                            
                            # 更新统计
                            with self._lock:
                                self.metrics['completed_tasks'] += 1
                                self.metrics['total_execution_time'] += task.execution_time
                                self.task_results[task.task_id] = result
                                
                            # 更新调度历史
                            gpu_info = next(g for g in gpus if g.index == allocated_gpu)
                            self.scheduler.update_history(task, gpu_info)
                            
                            break
                            
                    except Exception as e:
                        logger.error(f"Task {task.task_id} failed: {e}")
                        task.retry_count += 1
                        
                        if task.retry_count > task.max_retries:
                            task.status = TaskStatus.FAILED
                            task.error = e
                            with self._lock:
                                self.metrics['failed_tasks'] += 1
                                
                        time.sleep(1)
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                
    def _monitor_loop(self):
        """监控线程主循环"""
        while self.running:
            try:
                # 收集GPU状态
                gpus = self.resource_pool.get_all_gpu_info()
                timestamp = datetime.now()
                
                for gpu in gpus:
                    self.metrics['gpu_utilization_history'][gpu.index].append({
                        'timestamp': timestamp,
                        'utilization': gpu.utilization,
                        'memory_usage': gpu.memory_usage_percent,
                        'temperature': gpu.temperature,
                        'power_usage': gpu.power_usage
                    })
                    
                    # 保留最近1小时的数据
                    history = self.metrics['gpu_utilization_history'][gpu.index]
                    cutoff_time = timestamp - timedelta(hours=1)
                    self.metrics['gpu_utilization_history'][gpu.index] = [
                        h for h in history if h['timestamp'] > cutoff_time
                    ]
                    
                time.sleep(5)  # 5秒采样一次
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                
    def submit_task(self, 
                   func: Callable,
                   args: tuple = (),
                   kwargs: dict = None,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   required_memory_gb: float = 1.0,
                   timeout: Optional[float] = None,
                   callback: Optional[Callable] = None,
                   error_callback: Optional[Callable] = None) -> str:
        """
        提交GPU任务
        
        Args:
            func: 要执行的函数
            args: 函数参数
            kwargs: 函数关键字参数
            priority: 任务优先级
            required_memory_gb: 所需GPU内存(GB)
            timeout: 超时时间(秒)
            callback: 成功回调函数
            error_callback: 错误回调函数
            
        Returns:
            task_id: 任务ID
        """
        if kwargs is None:
            kwargs = {}
            
        # 创建任务
        task_id = f"task_{int(time.time()*1000)}_{np.random.randint(1000)}"
        task = GPUTask(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            required_memory_gb=required_memory_gb,
            timeout=timeout,
            callback=callback,
            error_callback=error_callback
        )
        
        # 加入队列
        priority_value = -task.priority.value  # 负值使高优先级先执行
        self.task_queue.put((priority_value, task))
        
        with self._lock:
            self.metrics['total_tasks'] += 1
            
        logger.info(f"Task {task_id} submitted with priority {priority.name}")
        return task_id
        
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        获取任务结果
        
        Args:
            task_id: 任务ID
            timeout: 超时时间
            
        Returns:
            任务执行结果
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                if task_id in self.task_results:
                    return self.task_results.pop(task_id)
                    
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} result timeout")
                
            time.sleep(0.1)
            
    def batch_process(self,
                     data: Union[List, np.ndarray],
                     process_func: Callable,
                     batch_size: Optional[int] = None,
                     priority: TaskPriority = TaskPriority.NORMAL,
                     required_memory_gb: float = 1.0) -> List[Any]:
        """
        批量处理数据
        
        Args:
            data: 要处理的数据
            process_func: 处理函数
            batch_size: 批次大小(None表示自动计算)
            priority: 任务优先级
            required_memory_gb: 每批次所需内存
            
        Returns:
            处理结果列表
        """
        # 自动计算批次大小
        if batch_size is None:
            gpus = self.resource_pool.get_all_gpu_info()
            if gpus:
                avg_memory = np.mean([g.available_memory_gb for g in gpus])
                batch_size = max(1, int(len(data) * required_memory_gb / avg_memory))
            else:
                batch_size = 1
                
        # 分批处理
        batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        task_ids = []
        
        for i, batch in enumerate(batches):
            task_id = self.submit_task(
                func=process_func,
                args=(batch,),
                priority=priority,
                required_memory_gb=required_memory_gb
            )
            task_ids.append(task_id)
            
        # 收集结果
        results = []
        for task_id in task_ids:
            result = self.get_task_result(task_id)
            results.extend(result if isinstance(result, list) else [result])
            
        return results
        
    def get_status(self) -> Dict:
        """获取GPU管理器状态"""
        with self._lock:
            gpus = self.resource_pool.get_all_gpu_info()
            
            return {
                'running': self.running,
                'workers': len(self.workers),
                'pending_tasks': self.task_queue.qsize(),
                'active_tasks': len(self.executor.current_tasks),
                'total_tasks': self.metrics['total_tasks'],
                'completed_tasks': self.metrics['completed_tasks'],
                'failed_tasks': self.metrics['failed_tasks'],
                'gpus': [asdict(gpu) for gpu in gpus],
                'current_process_usage': self.resource_pool.process_monitor.get_current_process_gpu_usage()
            }
    
    def print_status(self):
        """打印GPU状态信息"""
        status = self.get_status()
        
        print("\n" + "="*100)
        print(f"GPU资源管理器状态 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)
        
        # 基本信息
        print(f"\n运行状态: {'✅ 运行中' if status['running'] else '❌ 已停止'}")
        print(f"工作线程: {status['workers']}")
        print(f"待处理任务: {status['pending_tasks']}")
        print(f"执行中任务: {status['active_tasks']}")
        print(f"总任务数: {status['total_tasks']} (完成: {status['completed_tasks']}, 失败: {status['failed_tasks']})")
        
        # GPU信息
        print("\n" + "-"*100)
        print(f"{'GPU':<4} {'名称':<30} {'显存使用':<20} {'利用率':<10} {'温度':<8} {'功耗':<8} {'进程数':<8}")
        print("-"*100)
        
        for gpu in status['gpus']:
            mem_usage = f"{gpu['memory_used']/1024:.1f}/{gpu['memory_total']/1024:.1f}GB"
            print(f"{gpu['index']:<4} {gpu['name']:<30} {mem_usage:<20} "
                  f"{gpu['utilization']:.0f}%{'':<7} {gpu['temperature']:.0f}°C{'':<5} "
                  f"{gpu['power_usage']:.0f}W{'':<5} {len(gpu['processes']):<8}")
        
        # 当前进程GPU使用
        if status['current_process_usage']:
            print(f"\n当前进程GPU使用: {status['current_process_usage']}")
        
        print("="*100)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        
    def shutdown(self):
        """关闭GPU管理器"""
        if not self.running:
            return
            
        logger.info("Shutting down GPU Manager...")
        self.running = False
        self._shutdown_event.set()
        
        # 等待工作线程结束
        for worker in self.workers:
            worker.join(timeout=5)
            
        # 等待监控线程结束
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
        logger.info("GPU Manager shutdown complete")


class GPUWebUI:
    """GPU管理器Web界面"""
    
    def __init__(self, gpu_manager: GPUManager):
        self.gpu_manager = gpu_manager
        self.update_interval = 5
        
    def create_interface(self):
        """创建Gradio界面"""
        if not GRADIO_AVAILABLE:
            logger.error("Gradio not available. Cannot create web UI.")
            return None
            
        with gr.Blocks(title="GPU资源管理系统", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🚀 GPU资源管理系统")
            gr.Markdown("实时监控和管理GPU资源，智能调度计算任务")
            
            with gr.Tabs():
                # 状态监控标签
                with gr.TabItem("📊 状态监控"):
                    with gr.Row():
                        status_text = gr.JSON(label="系统状态")
                        refresh_btn = gr.Button("🔄 刷新", variant="primary")
                    
                    with gr.Row():
                        gpu_usage_plot = gr.Plot(label="GPU使用率趋势")
                        memory_usage_plot = gr.Plot(label="显存使用趋势")
                    
                    with gr.Row():
                        temperature_plot = gr.Plot(label="温度监控")
                        power_plot = gr.Plot(label="功耗监控")
                
                # 任务管理标签
                with gr.TabItem("📋 任务管理"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            task_func = gr.Textbox(
                                label="任务函数", 
                                placeholder="例如: numpy.dot",
                                value="numpy.random.randn"
                            )
                            task_args = gr.Textbox(
                                label="参数", 
                                placeholder="(1000, 1000)",
                                value="(1000, 1000)"
                            )
                            required_memory = gr.Slider(
                                label="所需显存(GB)", 
                                minimum=0.5, 
                                maximum=24, 
                                value=1.0,
                                step=0.5
                            )
                            priority = gr.Dropdown(
                                label="优先级",
                                choices=["CRITICAL", "HIGH", "NORMAL", "LOW", "IDLE"],
                                value="NORMAL"
                            )
                        
                        with gr.Column(scale=1):
                            submit_btn = gr.Button("提交任务", variant="primary")
                            task_id_output = gr.Textbox(label="任务ID")
                            task_result = gr.JSON(label="任务结果")
                    
                    task_list = gr.Dataframe(
                        headers=["任务ID", "状态", "GPU", "开始时间", "执行时间"],
                        label="任务列表"
                    )
                
                # 进程监控标签
                with gr.TabItem("🔍 进程监控"):
                    process_table = gr.Dataframe(
                        headers=["PID", "进程名", "用户", "GPU", "显存(MB)", "CPU%", "状态"],
                        label="GPU进程列表"
                    )
                    
                    with gr.Row():
                        kill_pid = gr.Number(label="PID", precision=0)
                        kill_btn = gr.Button("终止进程", variant="stop")
                
                # 配置管理标签
                with gr.TabItem("⚙️ 配置管理"):
                    with gr.Row():
                        with gr.Column():
                            min_memory_config = gr.Slider(
                                label="最小可用显存(GB)",
                                minimum=0.5,
                                maximum=8,
                                value=self.gpu_manager.resource_pool.min_free_memory_gb,
                                step=0.5
                            )
                            scheduling_strategy = gr.Dropdown(
                                label="调度策略",
                                choices=[s.value for s in SchedulingStrategy],
                                value=self.gpu_manager.scheduler.strategy.value
                            )
                            max_workers_config = gr.Slider(
                                label="最大工作线程数",
                                minimum=1,
                                maximum=16,
                                value=self.gpu_manager.max_workers,
                                step=1
                            )
                        
                        with gr.Column():
                            apply_config_btn = gr.Button("应用配置", variant="primary")
                            config_status = gr.Textbox(label="配置状态")
                            
                            export_metrics_btn = gr.Button("导出统计数据")
                            metrics_file = gr.File(label="统计数据文件")
            
            # 定义事件处理函数
            def refresh_status():
                return self.gpu_manager.get_status()
            
            def plot_gpu_usage():
                """绘制GPU使用率图表"""
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for gpu_idx, history in self.gpu_manager.metrics['gpu_utilization_history'].items():
                    if history:
                        timestamps = [h['timestamp'] for h in history[-100:]]
                        utilizations = [h['utilization'] for h in history[-100:]]
                        ax.plot(timestamps, utilizations, label=f'GPU {gpu_idx}')
                
                ax.set_xlabel('时间')
                ax.set_ylabel('使用率 (%)')
                ax.set_title('GPU使用率趋势')
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                return fig
            
            def plot_memory_usage():
                """绘制显存使用图表"""
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for gpu_idx, history in self.gpu_manager.metrics['gpu_utilization_history'].items():
                    if history:
                        timestamps = [h['timestamp'] for h in history[-100:]]
                        memory_usage = [h['memory_usage'] for h in history[-100:]]
                        ax.plot(timestamps, memory_usage, label=f'GPU {gpu_idx}')
                
                ax.set_xlabel('时间')
                ax.set_ylabel('显存使用 (%)')
                ax.set_title('显存使用趋势')
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                return fig
            
            def plot_temperature():
                """绘制温度图表"""
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for gpu_idx, history in self.gpu_manager.metrics['gpu_utilization_history'].items():
                    if history:
                        timestamps = [h['timestamp'] for h in history[-100:]]
                        temperatures = [h['temperature'] for h in history[-100:]]
                        ax.plot(timestamps, temperatures, label=f'GPU {gpu_idx}')
                
                ax.axhline(y=85, color='r', linestyle='--', label='警戒温度')
                ax.set_xlabel('时间')
                ax.set_ylabel('温度 (°C)')
                ax.set_title('GPU温度监控')
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                return fig
            
            def plot_power():
                """绘制功耗图表"""
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for gpu_idx, history in self.gpu_manager.metrics['gpu_utilization_history'].items():
                    if history:
                        timestamps = [h['timestamp'] for h in history[-100:]]
                        power_usage = [h['power_usage'] for h in history[-100:]]
                        ax.plot(timestamps, power_usage, label=f'GPU {gpu_idx}')
                
                ax.set_xlabel('时间')
                ax.set_ylabel('功耗 (W)')
                ax.set_title('GPU功耗监控')
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                return fig
            
            def submit_task(func_name, args_str, memory, priority_str):
                """提交任务"""
                try:
                    # 解析函数
                    module_name, func_name = func_name.rsplit('.', 1)
                    module = __import__(module_name)
                    func = getattr(module, func_name)
                    
                    # 解析参数
                    args = eval(args_str)
                    if not isinstance(args, tuple):
                        args = (args,)
                    
                    # 提交任务
                    priority = TaskPriority[priority_str]
                    task_id = self.gpu_manager.submit_task(
                        func=func,
                        args=args,
                        priority=priority,
                        required_memory_gb=memory
                    )
                    
                    return task_id, "任务提交成功"
                    
                except Exception as e:
                    return "", f"错误: {str(e)}"
            
            def get_task_list():
                """获取任务列表"""
                tasks = []
                
                # 获取当前执行的任务
                for task in self.gpu_manager.executor.current_tasks.values():
                    tasks.append([
                        task.task_id,
                        task.status.value,
                        task.assigned_gpu or "等待中",
                        task.start_time.strftime('%H:%M:%S') if task.start_time else "",
                        str(task.execution_time) if task.execution_time else ""
                    ])
                
                return tasks
            
            def get_process_list():
                """获取进程列表"""
                processes = []
                gpus = self.gpu_manager.resource_pool.get_all_gpu_info()
                
                for gpu in gpus:
                    for proc in gpu.processes:
                        processes.append([
                            proc['pid'],
                            proc['name'],
                            proc['username'],
                            gpu.index,
                            proc['memory_mb'],
                            f"{proc['cpu_percent']:.1f}",
                            proc['status']
                        ])
                
                return processes
            
            def kill_process(pid):
                """终止进程"""
                try:
                    if pid:
                        os.kill(int(pid), signal.SIGTERM)
                        return f"已发送终止信号到进程 {pid}"
                    return "请输入PID"
                except Exception as e:
                    return f"错误: {str(e)}"
            
            def apply_config(min_mem, strategy, max_workers):
                """应用配置"""
                try:
                    self.gpu_manager.resource_pool.min_free_memory_gb = min_mem
                    self.gpu_manager.scheduler.strategy = SchedulingStrategy(strategy)
                    # 注意：max_workers需要重启才能生效
                    return "配置已更新（部分配置需要重启生效）"
                except Exception as e:
                    return f"配置更新失败: {str(e)}"
            
            def export_metrics():
                """导出统计数据"""
                try:
                    import json
                    
                    metrics_data = {
                        'export_time': datetime.now().isoformat(),
                        'metrics': self.gpu_manager.metrics,
                        'status': self.gpu_manager.get_status()
                    }
                    
                    filename = f"gpu_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w') as f:
                        json.dump(metrics_data, f, indent=2, default=str)
                    
                    return filename
                except Exception as e:
                    logger.error(f"Export metrics failed: {e}")
                    return None
            
            # 绑定事件
            refresh_btn.click(
                fn=refresh_status,
                outputs=[status_text]
            )
            
            refresh_btn.click(
                fn=plot_gpu_usage,
                outputs=[gpu_usage_plot]
            )
            
            refresh_btn.click(
                fn=plot_memory_usage,
                outputs=[memory_usage_plot]
            )
            
            refresh_btn.click(
                fn=plot_temperature,
                outputs=[temperature_plot]
            )
            
            refresh_btn.click(
                fn=plot_power,
                outputs=[power_plot]
            )
            
            submit_btn.click(
                fn=submit_task,
                inputs=[task_func, task_args, required_memory, priority],
                outputs=[task_id_output, task_result]
            )
            
            refresh_btn.click(
                fn=get_task_list,
                outputs=[task_list]
            )
            
            refresh_btn.click(
                fn=get_process_list,
                outputs=[process_table]
            )
            
            kill_btn.click(
                fn=kill_process,
                inputs=[kill_pid],
                outputs=[config_status]
            )
            
            apply_config_btn.click(
                fn=apply_config,
                inputs=[min_memory_config, scheduling_strategy, max_workers_config],
                outputs=[config_status]
            )
            
            export_metrics_btn.click(
                fn=export_metrics,
                outputs=[metrics_file]
            )
            
            # 自动更新
            interface.load(refresh_status, outputs=[status_text])
            
        return interface


# 便捷API函数
_default_manager = None

def init_gpu_manager(**kwargs) -> GPUManager:
    """初始化默认GPU管理器"""
    global _default_manager
    _default_manager = GPUManager(**kwargs)
    _default_manager.start()
    return _default_manager

def get_gpu_manager() -> GPUManager:
    """获取默认GPU管理器"""
    global _default_manager
    if _default_manager is None:
        _default_manager = init_gpu_manager()
    return _default_manager

def gpu_task(required_memory_gb: float = 1.0, 
             priority: TaskPriority = TaskPriority.NORMAL,
             timeout: Optional[float] = None):
    """GPU任务装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_gpu_manager()
            task_id = manager.submit_task(
                func=func,
                args=args,
                kwargs=kwargs,
                required_memory_gb=required_memory_gb,
                priority=priority,
                timeout=timeout
            )
            return manager.get_task_result(task_id)
        return wrapper
    return decorator


# 示例用法
if __name__ == "__main__":
    # 初始化GPU管理器
    gpu_manager = GPUManager(
        min_free_memory_gb=2.0,
        scheduling_strategy=SchedulingStrategy.ADAPTIVE,
        max_workers=4
    )
    
    # 启动管理器
    gpu_manager.start()
    
    # 打印状态
    gpu_manager.print_status()
    
    # 示例：使用装饰器
    @gpu_task(required_memory_gb=1.0, priority=TaskPriority.HIGH)
    def matrix_multiply(size=1000):
        import numpy as np
        a = np.random.randn(size, size)
        b = np.random.randn(size, size)
        return np.dot(a, b)
    
    # 执行任务
    # result = matrix_multiply(2000)
    
    # 批量处理示例
    # data = list(range(100))
    # results = gpu_manager.batch_process(
    #     data=data,
    #     process_func=lambda x: [i**2 for i in x],
    #     batch_size=10
    # )
    
    # 启动Web界面
    if GRADIO_AVAILABLE:
        web_ui = GPUWebUI(gpu_manager)
        interface = web_ui.create_interface()
        if interface:
            interface.launch(server_name="0.0.0.0", server_port=7860, share=True)
    
    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        gpu_manager.shutdown()
            