#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Training Examples with GPU Resource Manager
Demonstrates integration with popular ML frameworks
"""

import numpy as np
import time
from gpu_manager import GPUManager, gpu_task, TaskPriority, SchedulingStrategy

# Optional imports (install as needed)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

try:
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. Install with: pip install scikit-learn")


def example_xgboost_training():
    """Example: XGBoost model training with GPU acceleration"""
    if not XGBOOST_AVAILABLE or not SKLEARN_AVAILABLE:
        print("Skipping XGBoost example (dependencies not available)")
        return
    
    print("\n=== XGBoost GPU Training Example ===")
    
    # Initialize GPU manager
    manager = GPUManager(
        min_free_memory_gb=2.0,
        scheduling_strategy=SchedulingStrategy.MEMORY_FIRST
    )
    manager.start()
    
    # Generate dataset
    print("Generating dataset...")
    X, y = make_classification(
        n_samples=100000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        n_clusters_per_class=3,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    @gpu_task(required_memory_gb=4.0, priority=TaskPriority.HIGH)
    def train_xgboost_model(X_train, y_train, X_test, y_test, params):
        """Train XGBoost model on GPU"""
        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Set GPU parameters
        gpu_params = params.copy()
        gpu_params.update({
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': 0  # Will be set by GPU manager
        })
        
        # Train model
        start_time = time.time()
        model = xgb.train(
            gpu_params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=20
        )
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(dtest)
        y_pred_binary = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred_binary)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'training_time': training_time,
            'best_iteration': model.best_iteration
        }
    
    # Define hyperparameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # Train model
    print("Training XGBoost model on GPU...")
    result = train_xgboost_model(X_train, y_train, X_test, y_test, params)
    
    print(f"\n✅ Training completed!")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Training time: {result['training_time']:.2f} seconds")
    print(f"Best iteration: {result['best_iteration']}")
    
    manager.shutdown()


def example_hyperparameter_tuning():
    """Example: Distributed hyperparameter tuning with GPU"""
    if not XGBOOST_AVAILABLE or not SKLEARN_AVAILABLE:
        print("Skipping hyperparameter tuning example (dependencies not available)")
        return
    
    print("\n=== Hyperparameter Tuning Example ===")
    
    # Initialize GPU manager with multiple workers
    manager = GPUManager(
        min_free_memory_gb=2.0,
        scheduling_strategy=SchedulingStrategy.ADAPTIVE,
        max_workers=4
    )
    manager.start()
    
    # Generate dataset
    print("Generating dataset...")
    X, y = make_regression(
        n_samples=50000,
        n_features=20,
        n_informative=15,
        noise=0.1,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define parameter grid
    param_grid = [
        {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100},
        {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 100},
        {'max_depth': 7, 'learning_rate': 0.1, 'n_estimators': 100},
        {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 200},
        {'max_depth': 5, 'learning_rate': 0.2, 'n_estimators': 50},
        {'max_depth': 6, 'learning_rate': 0.15, 'n_estimators': 80},
    ]
    
    def train_with_params(params_dict):
        """Train model with specific parameters"""
        import xgboost as xgb
        from sklearn.metrics import mean_squared_error
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Update params for GPU
        gpu_params = {
            'objective': 'reg:squarederror',
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            **params_dict
        }
        
        # Train
        model = xgb.train(
            gpu_params,
            dtrain,
            num_boost_round=params_dict['n_estimators'],
            evals=[(dtest, 'test')],
            verbose_eval=False
        )
        
        # Evaluate
        y_pred = model.predict(dtest)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'params': params_dict,
            'rmse': rmse,
            'model': model
        }
    
    # Submit all parameter combinations
    print(f"Testing {len(param_grid)} parameter combinations...")
    task_ids = []
    
    for params in param_grid:
        task_id = manager.submit_task(
            func=train_with_params,
            args=(params,),
            required_memory_gb=2.0,
            priority=TaskPriority.NORMAL
        )
        task_ids.append(task_id)
    
    # Collect results
    results = []
    for task_id in task_ids:
        result = manager.get_task_result(task_id)
        results.append(result)
        print(f"Params: {result['params']} -> RMSE: {result['rmse']:.4f}")
    
    # Find best parameters
    best_result = min(results, key=lambda x: x['rmse'])
    print(f"\n🏆 Best parameters: {best_result['params']}")
    print(f"Best RMSE: {best_result['rmse']:.4f}")
    
    manager.shutdown()


def example_ensemble_training():
    """Example: Training ensemble models in parallel"""
    if not SKLEARN_AVAILABLE:
        print("Skipping ensemble example (scikit-learn not available)")
        return
    
    print("\n=== Ensemble Model Training Example ===")
    
    manager = GPUManager(
        min_free_memory_gb=1.0,
        scheduling_strategy=SchedulingStrategy.LEAST_LOADED,
        max_workers=6
    )
    manager.start()
    
    # Generate dataset
    X, y = make_classification(
        n_samples=10000,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define different model configurations
    model_configs = [
        {
            'name': 'XGBoost-1',
            'type': 'xgboost',
            'params': {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 100}
        },
        {
            'name': 'XGBoost-2',
            'type': 'xgboost',
            'params': {'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 150}
        },
        {
            'name': 'LightGBM-1',
            'type': 'lightgbm',
            'params': {'num_leaves': 31, 'learning_rate': 0.1, 'n_estimators': 100}
        },
        {
            'name': 'LightGBM-2',
            'type': 'lightgbm',
            'params': {'num_leaves': 50, 'learning_rate': 0.05, 'n_estimators': 150}
        },
    ]
    
    @gpu_task(required_memory_gb=2.0, priority=TaskPriority.NORMAL)
    def train_ensemble_member(config, X_train, y_train, X_test, y_test):
        """Train a single ensemble member"""
        start_time = time.time()
        
        if config['type'] == 'xgboost' and XGBOOST_AVAILABLE:
            import xgboost as xgb
            params = config['params'].copy()
            params.update({
                'objective': 'multi:softprob',
                'num_class': 3,
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor'
            })
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=params['n_estimators'],
                verbose_eval=False
            )
            
            y_pred_proba = model.predict(dtest)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
        elif config['type'] == 'lightgbm' and LIGHTGBM_AVAILABLE:
            import lightgbm as lgb
            params = config['params'].copy()
            params.update({
                'objective': 'multiclass',
                'num_class': 3,
                'device': 'gpu',
                'gpu_use_dp': True
            })
            
            train_data = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(
                params,
                train_data,
                num_boost_round=params['n_estimators'],
                valid_sets=[train_data],
                callbacks=[lgb.log_evaluation(0)]
            )
            
            y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            # Fallback to CPU random forest
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**config['params'], n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        training_time = time.time() - start_time
        
        return {
            'name': config['name'],
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy,
            'training_time': training_time
        }
    
    # Train all models in parallel
    print("Training ensemble members in parallel...")
    ensemble_results = []
    
    for config in model_configs:
        result = train_ensemble_member(config, X_train, y_train, X_test, y_test)
        ensemble_results.append(result)
        print(f"{config['name']}: Accuracy={result['accuracy']:.4f}, Time={result['training_time']:.2f}s")
    
    # Combine predictions (simple voting)
    print("\nCombining ensemble predictions...")
    all_predictions = np.array([r['predictions'] for r in ensemble_results])
    ensemble_pred = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), 
        axis=0, 
        arr=all_predictions
    )
    
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    print(f"\n🎯 Ensemble accuracy: {ensemble_accuracy:.4f}")
    
    # Compare with individual models
    print("\nIndividual model accuracies:")
    for result in ensemble_results:
        print(f"  {result['name']}: {result['accuracy']:.4f}")
    
    manager.shutdown()


def example_batch_prediction():
    """Example: Batch prediction with large datasets"""
    print("\n=== Batch Prediction Example ===")
    
    manager = GPUManager(
        min_free_memory_gb=1.0,
        scheduling_strategy=SchedulingStrategy.MEMORY_FIRST
    )
    manager.start()
    
    # Create a trained model (simulated)
    print("Creating model and test data...")
    n_samples = 1000000
    n_features = 50
    
    # Simulate a trained model
    @gpu_task(required_memory_gb=2.0)
    def create_model():
        if XGBOOST_AVAILABLE:
            import xgboost as xgb
            # Create dummy training data
            X_dummy = np.random.randn(1000, n_features)
            y_dummy = np.random.randint(0, 2, 1000)
            
            dtrain = xgb.DMatrix(X_dummy, label=y_dummy)
            params = {
                'objective': 'binary:logistic',
                'tree_method': 'gpu_hist',
                'max_depth': 3,
                'learning_rate': 0.1
            }
            model = xgb.train(params, dtrain, num_boost_round=10)
            return model
        else:
            # Return dummy model
            return lambda x: np.random.rand(len(x))
    
    model = create_model()
    
    # Generate large test dataset
    test_data = np.random.randn(n_samples, n_features).astype(np.float32)
    print(f"Generated test data: {test_data.shape}")
    
    # Define batch prediction function
    def predict_batch(batch_data):
        if XGBOOST_AVAILABLE and hasattr(model, 'predict'):
            import xgboost as xgb
            dtest = xgb.DMatrix(batch_data)
            predictions = model.predict(dtest)
        else:
            predictions = np.random.rand(len(batch_data))
        return predictions
    
    # Perform batch predictions
    print("\nPerforming batch predictions...")
    start_time = time.time()
    
    predictions = manager.batch_process(
        data=test_data,
        process_func=predict_batch,
        batch_size=50000,  # Process 50k samples at a time
        required_memory_gb=1.0
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ Predictions completed!")
    print(f"Total samples: {len(predictions)}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Throughput: {len(predictions)/elapsed_time:.0f} samples/second")
    
    # Analyze predictions
    predictions_array = np.array(predictions)
    print(f"\nPrediction statistics:")
    print(f"  Mean: {np.mean(predictions_array):.4f}")
    print(f"  Std: {np.std(predictions_array):.4f}")
    print(f"  Min: {np.min(predictions_array):.4f}")
    print(f"  Max: {np.max(predictions_array):.4f}")
    
    manager.shutdown()


def example_real_time_inference():
    """Example: Real-time inference server with GPU"""
    print("\n=== Real-time Inference Example ===")
    
    manager = GPUManager(
        min_free_memory_gb=1.0,
        scheduling_strategy=SchedulingStrategy.LEAST_LOADED,
        max_workers=2
    )
    manager.start()
    
    # Simulate model loading
    @gpu_task(required_memory_gb=1.0)
    def load_model():
        print("Loading model into GPU memory...")
        # Simulate model loading
        time.sleep(1)
        return "model_loaded"
    
    model = load_model()
    
    # Simulate real-time requests
    print("\nSimulating real-time inference requests...")
    
    def process_request(request_id, data_size):
        """Process a single inference request"""
        import cupy as cp
        
        # Simulate data preprocessing
        data = cp.random.randn(data_size, 100, dtype=cp.float32)
        
        # Simulate inference
        start = time.time()
        result = cp.sum(data, axis=1)  # Simplified computation
        inference_time = (time.time() - start) * 1000  # ms
        
        return {
            'request_id': request_id,
            'result_shape': result.shape,
            'inference_time_ms': inference_time
        }
    
    # Submit multiple concurrent requests
    request_times = []
    num_requests = 20
    
    print(f"Submitting {num_requests} inference requests...")
    for i in range(num_requests):
        start_time = time.time()
        
        # Submit with varying data sizes
        data_size = np.random.randint(100, 1000)
        task_id = manager.submit_task(
            func=process_request,
            args=(f"req_{i}", data_size),
            required_memory_gb=0.5,
            priority=TaskPriority.HIGH,
            timeout=5.0
        )
        
        # Get result
        result = manager.get_task_result(task_id, timeout=10.0)
        total_time = (time.time() - start_time) * 1000  # ms
        
        request_times.append(total_time)
        print(f"Request {i}: Inference={result['inference_time_ms']:.1f}ms, "
              f"Total={total_time:.1f}ms")
        
        # Simulate request arrival pattern
        time.sleep(np.random.exponential(0.1))  # Poisson process
    
    # Statistics
    print(f"\n📊 Inference statistics:")
    print(f"  Average total time: {np.mean(request_times):.1f}ms")
    print(f"  P50 latency: {np.percentile(request_times, 50):.1f}ms")
    print(f"  P95 latency: {np.percentile(request_times, 95):.1f}ms")
    print(f"  P99 latency: {np.percentile(request_times, 99):.1f}ms")
    
    manager.shutdown()


if __name__ == "__main__":
    examples = [
        ("XGBoost Training", example_xgboost_training),
        ("Hyperparameter Tuning", example_hyperparameter_tuning),
        ("Ensemble Training", example_ensemble_training),
        ("Batch Prediction", example_batch_prediction),
        ("Real-time Inference", example_real_time_inference)
    ]
    
    print("🚀 GPU Resource Manager - Machine Learning Examples")
    print("=" * 60)
    
    for name, example_func in examples:
        try:
            example_func()
            time.sleep(2)
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ All ML examples completed!")