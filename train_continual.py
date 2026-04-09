"""
Continual Learning Training Script for RNN Models

This script implements continual learning where tasks are trained sequentially,
with each new task starting from the weights learned from previous tasks.
"""

import numpy as np
import torch
from model.model_wrapper import ModelWrapper, OptimizationParameters
from model.pt_models import VanillaArchitecture, Rank2Architechture
from data.data_generator import FamilyOfTasksGenerator
from data.functions import X, X2, X2Rotate, X4, X4Rotate, XReverse
from analysis.spectrum_analysis import SpectrumAnalysis
from analysis.tasks_pca import TasksPCA
import tools.pytorchtools as pytorchtools


def set_training_device(device_type='auto'):
    """
    Set the training device (MPS, CPU, or auto-detect).
    
    Args:
        device_type: 'mps', 'cpu', or 'auto' (default: 'auto')
                    - 'mps': Force use MPS (Apple Silicon GPU)
                    - 'cpu': Force use CPU
                    - 'auto': Automatically select best available device
    """
    if device_type == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            pytorchtools.device = torch.device("mps")
            print("Device set to: MPS (Apple Silicon GPU)")
        else:
            print("Warning: MPS not available, falling back to CPU")
            pytorchtools.device = torch.device("cpu")
    elif device_type == 'cpu':
        pytorchtools.device = torch.device("cpu")
        print("Device set to: CPU")
    else:  # 'auto'
        # Use the default device selection logic
        pytorchtools.device = pytorchtools.get_device()
        print(f"Device auto-selected: {pytorchtools.device}")


class ContinualLearningTrainer:
    """
    Trainer for continual learning scenarios where tasks are learned sequentially.
    """
    
    def __init__(self, architecture_func, units, tasks_list, instance_range, 
                 optimization_params=None, device='auto', **kwargs):
        """
        Args:
            architecture_func: RNN architecture class (e.g., VanillaArchitecture)
            units: Number of hidden units
            tasks_list: List of task lists, each list represents a learning stage
                        e.g., [[X()], [X(), X2()], [X(), X2(), X4()]]
            instance_range: Range of model instances to train
            optimization_params: Training parameters
            device: Device to use for training ('mps', 'cpu', or 'auto')
            **kwargs: Additional arguments for ModelWrapper
        """
        # Set training device before creating models
        set_training_device(device)
        
        self.architecture_func = architecture_func
        self.units = units
        self.tasks_list = tasks_list
        self.instance_range = instance_range
        self.optimization_params = optimization_params or OptimizationParameters()
        self.kwargs = kwargs
        self.device = device
        
        # Store wrappers for each stage
        self.stage_wrappers = []
        self.stage_weights = {}  # {stage: {instance: weights}}
        
    def train_stage(self, stage_idx, tasks, initial_weights=None):
        """
        Train on a specific stage of tasks.
        
        Args:
            stage_idx: Index of the current learning stage
            tasks: List of tasks for this stage
            initial_weights: Weights to initialize from (from previous stage)
        """
        print(f"\n{'='*60}")
        print(f"Training Stage {stage_idx + 1}/{len(self.tasks_list)}")
        print(f"Tasks: {[task.name for task in tasks]}")
        print(f"{'='*60}\n")
        
        # Create data generator for this stage
        train_data = FamilyOfTasksGenerator(
            tasks,
            input_type=self.kwargs.get('input_type', 'multi'),
            output_type=self.kwargs.get('output_type', 'multiple'),
            steps=self.kwargs.get('steps', 250),
            train_last=len(tasks)  # Train on all tasks in this stage
        )
        
        # Create model wrapper
        wrapper = ModelWrapper(
            architecture_func=self.architecture_func,
            units=self.units,
            train_data=train_data,
            instance_range=self.instance_range,
            optimization_params=self.optimization_params,
            **{k: v for k, v in self.kwargs.items() 
               if k not in ['input_type', 'output_type', 'steps']}
        )
        
        # Prepare initial weights
        weights = None
        if initial_weights is not None:
            weights = initial_weights
        
        # Train
        wrapper.train_model(weights=weights)
        
        # Save weights for next stage (in memory)
        self.stage_weights[stage_idx] = wrapper.get_all_weights()
        self.stage_wrappers.append(wrapper)
        
        # Print model save location
        print(f"\nStage {stage_idx + 1} models saved to:")
        for inst in self.instance_range:
            model_path = f"models/{wrapper.name}/i{inst}"
            print(f"  Instance {inst}: {model_path}/weights.pt")
        
        return wrapper
    
    def train_all_stages(self):
        """
        Train on all stages sequentially (continual learning).
        """
        initial_weights = None
        
        for stage_idx, tasks in enumerate(self.tasks_list):
            wrapper = self.train_stage(stage_idx, tasks, initial_weights)
            
            # Use weights from current stage as initial weights for next stage
            initial_weights = self.stage_weights[stage_idx]
        
        return self.stage_wrappers
    
    def evaluate_all_stages(self, analyses=None):
        """
        Evaluate model performance on all stages after continual learning.
        
        Args:
            analyses: List of analysis classes to run
        """
        if analyses is None:
            analyses = []
        
        results = {}
        for stage_idx, wrapper in enumerate(self.stage_wrappers):
            print(f"\nEvaluating Stage {stage_idx + 1}...")
            if analyses:
                wrapper.analyze(analyses)
            results[stage_idx] = wrapper
        
        return results
    
    def get_final_weights(self):
        """Get weights from the final training stage."""
        if self.stage_weights:
            final_stage = max(self.stage_weights.keys())
            return self.stage_weights[final_stage]
        return None
    
    def get_stage_model_paths(self, stage_idx):
        """
        Get the file paths where models for a specific stage are saved.
        
        Args:
            stage_idx: Index of the stage (0-based)
            
        Returns:
            dict: {instance: model_path} mapping
        """
        if stage_idx >= len(self.stage_wrappers):
            return {}
        
        wrapper = self.stage_wrappers[stage_idx]
        paths = {}
        for inst in self.instance_range:
            paths[inst] = f"models/{wrapper.name}/i{inst}"
        return paths
    
    def list_all_stage_models(self):
        """
        Print information about all saved stage models.
        """
        print("\n" + "="*60)
        print("Continual Learning Model Checkpoints")
        print("="*60)
        for stage_idx, wrapper in enumerate(self.stage_wrappers):
            print(f"\nStage {stage_idx + 1}: {[task.name for task in self.tasks_list[stage_idx]]}")
            print(f"  Model name: {wrapper.name}")
            for inst in self.instance_range:
                model_path = f"models/{wrapper.name}/i{inst}"
                print(f"    Instance {inst}: {model_path}/weights.pt")
        print("="*60 + "\n")


def continual_learning_example(device='auto'):
    """
    Example: Continual learning with increasing task complexity.
    
    Args:
        device: Device to use ('mps', 'cpu', or 'auto')
    """
    # Define learning stages - tasks are added incrementally
    # stages = [
    #     [X()],                          # Stage 1: Learn X
    #     [X2()],                    # Stage 2: Add X2
    #     [X2Rotate()],       # Stage 3: Add X2Rotate
    #     [X4()], # Stage 4: Add X4
    # ]

    stages = [
        [X()],                          # Stage 1: Learn X
        [X2()],                    # Stage 2: Add X2
    ]
    
    trainer = ContinualLearningTrainer(
        architecture_func=VanillaArchitecture,
        units=100,
        tasks_list=stages,
        instance_range=range(0, 1),
        optimization_params=OptimizationParameters(
            batch_size=32,
            epochs=100,
            minimal_loss=1e-4,
            initial_lr=1e-4
        ),
        device=device,
        recurrent_bias=False,
        readout_bias=True
    )
    
    # Train all stages
    wrappers = trainer.train_all_stages()
    
    # Evaluate
    trainer.evaluate_all_stages(analyses=[SpectrumAnalysis, TasksPCA])
    
    return trainer, wrappers


def continual_learning_rank2_example(device='auto'):
    """
    Example: Continual learning with Rank2 architecture on 6 tasks.
    
    Args:
        device: Device to use ('mps', 'cpu', or 'auto')
    """
    stages = [
        [X()],
        [X(), X2()],
        [X(), X2(), X2Rotate()],
        [X(), X2(), X2Rotate(), XReverse()],
        [X(), X2(), X2Rotate(), XReverse(), X4()],
        [X(), X2(), X2Rotate(), XReverse(), X4(), X4Rotate()],
    ]
    
    trainer = ContinualLearningTrainer(
        architecture_func=Rank2Architechture,
        units=100,
        tasks_list=stages,
        instance_range=range(0, 3),
        optimization_params=OptimizationParameters(
            batch_size=32,
            epochs=5000,
            minimal_loss=1e-4,
            initial_lr=1e-4
        ),
        device=device,
        recurrent_bias=True,
        readout_bias=True
    )
    
    wrappers = trainer.train_all_stages()
    trainer.evaluate_all_stages(analyses=[TasksPCA])
    
    return trainer, wrappers


class ContinualLearningWithReplay(ContinualLearningTrainer):
    """
    Enhanced continual learning with replay buffer to prevent catastrophic forgetting.
    Mixes data from previous tasks with new task data during training.
    """
    
    def __init__(self, replay_ratio=0.3, device='auto', *args, **kwargs):
        """
        Args:
            replay_ratio: Fraction of training data that comes from previous tasks
            device: Device to use ('mps', 'cpu', or 'auto')
        """
        # Extract device from kwargs if passed there, otherwise use parameter
        if 'device' in kwargs:
            device = kwargs.pop('device')
        super().__init__(*args, device=device, **kwargs)
        self.replay_ratio = replay_ratio
        self.replay_data = {}  # Store data from previous stages
    
    def train_stage(self, stage_idx, tasks, initial_weights=None):
        """
        Train on a stage with replay from previous tasks.
        """
        print(f"\n{'='*60}")
        print(f"Training Stage {stage_idx + 1}/{len(self.tasks_list)} (with replay)")
        print(f"Tasks: {[task.name for task in tasks]}")
        print(f"Replay ratio: {self.replay_ratio}")
        print(f"{'='*60}\n")
        
        # Create data generator for current tasks
        train_data = FamilyOfTasksGenerator(
            tasks,
            input_type=self.kwargs.get('input_type', 'multi'),
            output_type=self.kwargs.get('output_type', 'multiple'),
            steps=self.kwargs.get('steps', 250),
            train_last=len(tasks)
        )
        
        # If we have previous stages, create a mixed data generator
        if stage_idx > 0 and self.replay_ratio > 0:
            # Store current stage data for future replay
            x_new, y_new = train_data.generate_train_data()
            
            # Get replay data from previous stages
            replay_x_list = []
            replay_y_list = []
            for prev_stage in range(stage_idx):
                if prev_stage in self.replay_data:
                    x_rep, y_rep = self.replay_data[prev_stage]
                    n_replay = int(len(x_new) * self.replay_ratio / stage_idx)
                    indices = np.random.choice(len(x_rep), min(n_replay, len(x_rep)), replace=False)
                    replay_x_list.append(x_rep[indices])
                    replay_y_list.append(y_rep[indices])
            
            if replay_x_list:
                replay_x = np.concatenate(replay_x_list, axis=0)
                replay_y = np.concatenate(replay_y_list, axis=0)
                
                # Mix new and replay data
                n_new = int(len(x_new) * (1 - self.replay_ratio))
                indices = np.random.choice(len(x_new), n_new, replace=False)
                x_mixed = np.concatenate([x_new[indices], replay_x], axis=0)
                y_mixed = np.concatenate([y_new[indices], replay_y], axis=0)
                
                # Shuffle
                shuffle_idx = np.random.permutation(len(x_mixed))
                x_mixed = x_mixed[shuffle_idx]
                y_mixed = y_mixed[shuffle_idx]
                
                # Store for future replay
                self.replay_data[stage_idx] = (x_new, y_new)
                
                # Create wrapper and train with mixed data
                wrapper = ModelWrapper(
                    architecture_func=self.architecture_func,
                    units=self.units,
                    train_data=train_data,
                    instance_range=self.instance_range,
                    optimization_params=self.optimization_params,
                    **{k: v for k, v in self.kwargs.items() 
                       if k not in ['input_type', 'output_type', 'steps']}
                )
                
                weights = initial_weights
                wrapper.train_model_with_data(x_mixed, y_mixed, weights=weights)
            else:
                # First stage, no replay yet
                self.replay_data[stage_idx] = (x_new, y_new)
                wrapper = ModelWrapper(
                    architecture_func=self.architecture_func,
                    units=self.units,
                    train_data=train_data,
                    instance_range=self.instance_range,
                    optimization_params=self.optimization_params,
                    **{k: v for k, v in self.kwargs.items() 
                       if k not in ['input_type', 'output_type', 'steps']}
                )
                wrapper.train_model(weights=initial_weights)
        else:
            # First stage, no replay
            x_new, y_new = train_data.generate_train_data()
            self.replay_data[stage_idx] = (x_new, y_new)
            
            wrapper = ModelWrapper(
                architecture_func=self.architecture_func,
                units=self.units,
                train_data=train_data,
                instance_range=self.instance_range,
                optimization_params=self.optimization_params,
                **{k: v for k, v in self.kwargs.items() 
                   if k not in ['input_type', 'output_type', 'steps']}
            )
            wrapper.train_model(weights=initial_weights)
        
        # Save weights
        self.stage_weights[stage_idx] = wrapper.get_all_weights()
        self.stage_wrappers.append(wrapper)
        
        return wrapper


def continual_learning_with_replay_example(device='auto'):
    """
    Example: Continual learning with replay to prevent forgetting.
    
    Args:
        device: Device to use ('mps', 'cpu', or 'auto')
    """
    stages = [
        [X()],
        [X(), X2()],
        [X(), X2(), X2Rotate()],
        [X(), X2(), X2Rotate(), X4()],
    ]
    
    trainer = ContinualLearningWithReplay(
        architecture_func=VanillaArchitecture,
        units=100,
        tasks_list=stages,
        instance_range=range(0, 3),
        replay_ratio=0.3,  # 30% of data from previous tasks
        optimization_params=OptimizationParameters(
            batch_size=32,
            epochs=5000,
            minimal_loss=1e-4,
            initial_lr=1e-4
        ),
        device=device,
        recurrent_bias=False,
        readout_bias=True
    )
    
    wrappers = trainer.train_all_stages()
    trainer.evaluate_all_stages(analyses=[TasksPCA])
    
    return trainer, wrappers


if __name__ == '__main__':
    import sys
    
    print("Starting Continual Learning Training...")
    print("\nAvailable modes:")
    print("1. Basic continual learning (sequential task learning)")
    print("2. Continual learning with replay (prevents forgetting)")
    print("3. Rank2 continual learning")
    print("\nDevice options:")
    print("  - 'mps': Use Apple Silicon GPU (MPS)")
    print("  - 'cpu': Use CPU only")
    print("  - 'auto': Auto-detect best device (default)")
    
    # Parse command line arguments
    mode = '1'
    device = 'cpu'
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    if len(sys.argv) > 2:
        device = sys.argv[2].lower()
        if device not in ['mps', 'cpu', 'auto']:
            print(f"Warning: Unknown device '{device}', using 'auto'")
            device = 'auto'
    
    print(f"\nMode: {mode}, Device: {device}\n")
    
    if mode == '1':
        trainer, wrappers = continual_learning_example(device=device)
    elif mode == '2':
        trainer, wrappers = continual_learning_with_replay_example(device=device)
    elif mode == '3':
        trainer, wrappers = continual_learning_rank2_example(device=device)
    else:
        print(f"Unknown mode: {mode}, using basic continual learning")
        trainer, wrappers = continual_learning_example(device=device)
    
    print("\nContinual learning training completed!")
    print(f"Trained {len(wrappers)} stages")
    print(f"Final stage has {len(trainer.tasks_list[-1])} tasks")
    print(f"Training device: {pytorchtools.device}")

