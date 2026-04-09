"""
Plot training loss curves for Task A and Task B during continual learning.

This script visualizes the training loss for each task across different stages
in a continual learning setting.
"""

import matplotlib.pyplot as plt
from tools.utils import load_pickle
import os

# Set matplotlib parameters
plt.rcParams.update({
    'font.family': 'Arial',
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7
})


def plot_continual_learning_loss(trainer, instance=0, save_path=None):
    """
    Plot training loss curves for Task A and Task B across stages.
    
    Args:
        trainer: SwitchingContinualLearningTrainer instance
        instance: Model instance to plot
        save_path: Path to save the figure (optional)
    """
    # Handle None instance
    if instance is None:
        if hasattr(trainer, 'instance_range') and trainer.instance_range:
            instance = list(trainer.instance_range)[0]
        else:
            instance = 0
            print(f"Warning: instance is None, using default instance={instance}")
    
    stages = trainer.tasks_list
    active_task_indices = trainer.active_task_indices
    
    # Get task names
    task_names = []
    for stage in stages:
        stage_names = [task.name for task in stage]
        task_names.append(stage_names)
    
    # Collect loss data for each stage
    stage_losses = {0: [], 1: []}  # Task A and Task B losses
    stage_epochs = {0: [], 1: []}  # Epoch numbers for each task
    
    for stage_idx, wrapper in enumerate(trainer.stage_wrappers):
        model_path_base = f"models/{wrapper.name}"
        loss_file = f"{model_path_base}/i{instance}/loss_history.pkl"
        
        # Debug: print what we're looking for
        print(f"Stage {stage_idx + 1}: Looking for loss history at {loss_file}")
        
        # Try to load saved loss history
        if os.path.exists(loss_file):
            loss_history = load_pickle(loss_file)
            print(f"  ✓ Found loss history with {len(loss_history.get('losses', []))} epochs")
        else:
            print(f"  ✗ Loss history file not found at {loss_file}")
            print(f"Warning: No saved loss history for stage {stage_idx + 1}. "
                  f"Loss history needs to be saved during training.")
            continue
        
        active_task_idx = active_task_indices[stage_idx]
        
        # Extract loss data for this stage
        if 'task_losses' in loss_history:
            # If task-specific losses were saved
            task_losses = loss_history['task_losses']
            epochs = loss_history['epochs']
            
            if active_task_idx in task_losses:
                stage_losses[active_task_idx].extend(task_losses[active_task_idx])
                # Adjust epoch numbers to be cumulative
                if stage_epochs[active_task_idx]:
                    last_epoch = max(stage_epochs[active_task_idx])
                    stage_epochs[active_task_idx].extend([e + last_epoch for e in epochs])
                else:
                    stage_epochs[active_task_idx].extend(epochs)
        elif 'losses' in loss_history:
            # If only overall losses were saved
            losses = loss_history['losses']
            epochs = loss_history.get('epochs', range(len(losses)))
            
            if stage_epochs[active_task_idx]:
                last_epoch = max(stage_epochs[active_task_idx])
                stage_epochs[active_task_idx].extend([e + last_epoch for e in epochs])
            else:
                stage_epochs[active_task_idx].extend(epochs)
            
            stage_losses[active_task_idx].extend(losses)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    
    # Plot Task A losses
    if stage_epochs[0] and stage_losses[0]:
        ax.plot(stage_epochs[0], stage_losses[0], 'b-', label='Task A', linewidth=1.5, alpha=0.7)
    
    # Plot Task B losses
    if stage_epochs[1] and stage_losses[1]:
        ax.plot(stage_epochs[1], stage_losses[1], 'r-', label='Task B', linewidth=1.5, alpha=0.7)
    
    # Add vertical lines to indicate stage boundaries
    if stage_epochs[0] or stage_epochs[1]:
        all_epochs = []
        if stage_epochs[0]:
            all_epochs.extend(stage_epochs[0])
        if stage_epochs[1]:
            all_epochs.extend(stage_epochs[1])
        
        if all_epochs:
            max_epoch = max(all_epochs)
            for stage_idx in range(len(stages)):
                # Find the epoch where this stage ends
                stage_end = 0
                for task_idx in range(stage_idx + 1):
                    if stage_epochs[task_idx]:
                        task_epochs = [e for e in stage_epochs[task_idx] if e <= max_epoch]
                        if task_epochs:
                            stage_end = max(stage_end, max(task_epochs))
                
                if stage_end > 0:
                    ax.axvline(x=stage_end, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
                    ax.text(stage_end, ax.get_ylim()[1] * 0.95, f'Stage {stage_idx + 1}',
                           rotation=90, verticalalignment='top', fontsize=6)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Continual Learning: Training Loss for Task A and Task B')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.savefig('continual_learning_loss_curves.png', dpi=300, bbox_inches='tight')
        print("Figure saved to continual_learning_loss_curves.png")
    
    plt.show()


def plot_from_model_paths(model_paths, task_names, instance=0, save_path=None):
    """
    Plot loss curves directly from model paths.
    
    Args:
        model_paths: List of model paths for each stage
        task_names: List of task names for each stage (e.g., [['TaskA', 'TaskB'], ['TaskA', 'TaskB']])
        instance: Model instance to plot
        save_path: Path to save the figure (optional)
    """
    stage_losses = {}
    stage_epochs = {}
    active_task_indices = []
    
    # Determine active task indices from task names
    # For switching continual learning:
    # Stage 0: Task A (index 0) - e.g., flipflop2bits
    # Stage 1: Task B (index 1) - e.g., lines2bits
    for stage_idx, stage_tasks in enumerate(task_names):
        # For switching continual learning, each stage trains one task
        # Stage 0 -> Task 0, Stage 1 -> Task 1, etc.
        active_task_idx = stage_idx % len(stage_tasks) if stage_tasks else stage_idx
        active_task_indices.append(active_task_idx)
        
        if active_task_idx not in stage_losses:
            stage_losses[active_task_idx] = []
            stage_epochs[active_task_idx] = []
    
    # Load loss history from each stage
    cumulative_epoch = 0
    for stage_idx, model_path in enumerate(model_paths):
        loss_file = f"{model_path}/i{instance}/loss_history.pkl"
        
        if os.path.exists(loss_file):
            loss_history = load_pickle(loss_file)
            losses = loss_history.get('losses', [])
            epochs = loss_history.get('epochs', list(range(1, len(losses) + 1)))
            
            active_task_idx = active_task_indices[stage_idx]
            
            # Get task name for display
            task_name = task_names[stage_idx][active_task_idx] if stage_idx < len(task_names) and active_task_idx < len(task_names[stage_idx]) else f"Task {active_task_idx}"
            
            print(f"  Stage {stage_idx}: Loading {len(losses)} loss values for {task_name} (Task index {active_task_idx})")
            
            # Adjust epochs to be cumulative
            adjusted_epochs = [e + cumulative_epoch for e in epochs]
            stage_losses[active_task_idx].extend(losses)
            stage_epochs[active_task_idx].extend(adjusted_epochs)
            
            cumulative_epoch = max(adjusted_epochs) if adjusted_epochs else cumulative_epoch
        else:
            print(f"Warning: Loss history not found for stage {stage_idx + 1} at {loss_file}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for task_idx in sorted(stage_losses.keys()):
        if stage_epochs[task_idx] and stage_losses[task_idx]:
            # Get task name from first stage
            if task_names and task_idx < len(task_names[0]):
                task_label = f"Task {task_names[0][task_idx]}"
            else:
                task_label = f"Task {task_idx}"
            
            ax.plot(stage_epochs[task_idx], stage_losses[task_idx], 
                   color=colors[task_idx % len(colors)], 
                   label=task_label, linewidth=1.5, alpha=0.8, marker='o', markersize=1.5)
    
    # Add vertical lines for stage boundaries
    if stage_epochs:
        all_epochs = []
        for epochs in stage_epochs.values():
            all_epochs.extend(epochs)
        
        if all_epochs:
            max_epoch = max(all_epochs)
            # Find stage boundaries
            stage_boundaries = []
            current_epoch = 0
            for stage_idx, model_path in enumerate(model_paths):
                loss_file = f"{model_path}/i{instance}/loss_history.pkl"
                if os.path.exists(loss_file):
                    loss_history = load_pickle(loss_file)
                    epochs = loss_history.get('epochs', [])
                    if epochs:
                        current_epoch += max(epochs)
                        stage_boundaries.append(current_epoch)
            
            for i, boundary in enumerate(stage_boundaries[:-1]):  # Don't draw line after last stage
                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
                ax.text(boundary, ax.get_ylim()[1] * 0.95, f'Stage {i + 1}→{i + 2}',
                       rotation=90, verticalalignment='top', fontsize=6, ha='right')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Continual Learning: Training Loss Across Stages', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.savefig('continual_learning_loss_curves.png', dpi=300, bbox_inches='tight')
        print("Figure saved to continual_learning_loss_curves.png")
    
    plt.show()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage: provide model paths
        # Example: python plot_continual_learning_loss.py model1 model2 ...
        model_paths = sys.argv[1:]
        task_names = [['TaskA', 'TaskB']] * len(model_paths)  # Adjust based on your setup
        plot_from_model_paths(model_paths, task_names, instance=0)
    else:
        # Example usage with trainer
        print("Usage options:")
        print("1. With trainer object:")
        print("   from train_continual_switching import switching_continual_learning_example")
        print("   trainer, wrappers = switching_continual_learning_example(device='cpu')")
        print("   plot_continual_learning_loss(trainer, instance=0)")
        print()
        print("2. With model paths:")
        print("   python plot_continual_learning_loss.py models/path1 models/path2 ...")
        print()
        print("3. Direct function call:")
        print("   plot_from_model_paths(['models/path1', 'models/path2'], [['TaskA', 'TaskB'], ['TaskA', 'TaskB']])")

