"""
Script to plot trajectories in kappa space with basin of attraction.
Shows example trials and colors them by which attractor they converge to.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import torch
from pathlib import Path as PathLib
import contourpy

from model.pt_models import Rank2Architechture
from model.model_wrapper import load_weights
from analysis.analyzer import RankAnalyzer
from data.custom_data_generator import (
    FlipFlopLineGenerator,
    FlipFlopCycleGenerator,
    LimitCycleLineGenerator,
    OrthogonalFlipFlopLineGenerator,
    OrthogonalFlipFlopCycleGenerator,
    OrthogonalCycleLineGenerator,
    ParallelFlipFlopLineGenerator,
    ParallelFlipFlopCycleGenerator,
    ParallelCycleLineGenerator
)
import tools.pytorchtools as pytorchtools

plt.rcParams.update({
    'font.family': 'Arial',
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7
})


def get_data_generator(task_combination, training_mode="standard", n_bits=2):
    """
    Get the appropriate data generator for the task combination.
    
    Args:
        task_combination: List of task names, e.g., ['flipflop', 'line']
        training_mode: 'standard', 'orthogonal', or 'parallel'
        n_bits: Number of bits for the tasks
        
    Returns:
        Data generator instance
    """
    tasks_sorted = sorted(task_combination)
    
    if len(tasks_sorted) == 2:
        if 'flipflop' in tasks_sorted and 'line' in tasks_sorted:
            if training_mode == "orthogonal":
                return OrthogonalFlipFlopLineGenerator(n_bits=n_bits)
            elif training_mode == "parallel":
                return ParallelFlipFlopLineGenerator(n_bits=n_bits)
            else:
                return FlipFlopLineGenerator(n_bits=n_bits)
        
        elif 'flipflop' in tasks_sorted and 'cycle' in tasks_sorted:
            if training_mode == "orthogonal":
                return OrthogonalFlipFlopCycleGenerator(n_bits=n_bits)
            elif training_mode == "parallel":
                return ParallelFlipFlopCycleGenerator(n_bits=n_bits)
            else:
                return FlipFlopCycleGenerator(n_bits=n_bits)
        
        elif 'cycle' in tasks_sorted and 'line' in tasks_sorted:
            if training_mode == "orthogonal":
                return OrthogonalCycleLineGenerator(n_bits=n_bits)
            elif training_mode == "parallel":
                return ParallelCycleLineGenerator(n_bits=n_bits)
            else:
                return LimitCycleLineGenerator(n_bits=n_bits)
    
    raise ValueError(f"Unsupported task combination: {task_combination}")


def find_model_path(task_combination, training_mode="standard", rank=2, instance=100):
    """
    Find model path matching the task combination and training mode.
    """
    models_dir = PathLib("models")
    if not models_dir.exists():
        return None
    
    tasks_sorted = sorted(task_combination)
    
    if len(tasks_sorted) == 2:
        if 'flipflop' in tasks_sorted and 'line' in tasks_sorted:
            base_pattern = "flipflopline"
        elif 'flipflop' in tasks_sorted and 'cycle' in tasks_sorted:
            base_pattern = "flipflopcycle"
        elif 'cycle' in tasks_sorted and 'line' in tasks_sorted:
            base_pattern = "cycleline"
        else:
            base_pattern = "".join(tasks_sorted)
    else:
        base_pattern = "".join(tasks_sorted)
    
    if training_mode == "orthogonal":
        pattern = f"{base_pattern}orthogonal" if "orthogonal" not in base_pattern else base_pattern
    elif training_mode == "parallel":
        pattern = f"{base_pattern}parallel" if "parallel" not in base_pattern else base_pattern
    else:
        pattern = base_pattern
    
    for model_folder in models_dir.iterdir():
        if model_folder.is_dir():
            model_name = model_folder.name.lower()
            all_tasks_match = all(task in model_name for task in tasks_sorted)
            if all_tasks_match and f"rank{rank}" in model_name:
                if training_mode == "orthogonal" and "orthogonal" not in model_name:
                    continue
                elif training_mode == "parallel" and "parallel" not in model_name:
                    continue
                elif training_mode == "standard" and ("orthogonal" in model_name or "parallel" in model_name):
                    continue
                
                instance_dir = model_folder / f"i{instance}"
                weights_file = instance_dir / "weights.pt"
                if weights_file.exists():
                    return str(weights_file)
    
    return None


def load_model_and_create_rank_analyzer(model_path, units=100, n_bits=2, steps=200):
    """
    Load model and create RankAnalyzer for kappa space operations.
    """
    weights = load_weights(PathLib(model_path).parent, "weights", map_location='cpu')
    
    # Extract m and n from weights (handle different key formats)
    possible_m_keys = ['rnn.m', 'm', 'rnn.rnn.m']
    possible_n_keys = ['rnn.n', 'n', 'rnn.rnn.n']
    
    m_tensor = None
    n_tensor = None
    
    for key in possible_m_keys:
        if key in weights:
            m_tensor = weights[key]
            if torch.is_tensor(m_tensor):
                m_tensor = m_tensor.cpu().numpy()
            break
    
    for key in possible_n_keys:
        if key in weights:
            n_tensor = weights[key]
            if torch.is_tensor(n_tensor):
                n_tensor = n_tensor.cpu().numpy()
            break
    
    if m_tensor is None or n_tensor is None:
        print(f"Available weight keys: {list(weights.keys())[:20]}...")
        raise ValueError("Could not extract 'm' and 'n' from model weights")
    
    # Create a simple model wrapper to use RankAnalyzer
    # RankAnalyzer needs: 'm', 'n', 'Who.weight', 'Wih.weight', optionally 'Wih.bias'
    class SimpleModel:
        def __init__(self, weights_dict, m, n):
            # Create normalized weights dict for RankAnalyzer
            self.weights = {}
            
            # Add m and n
            if isinstance(m, np.ndarray):
                self.weights['m'] = torch.from_numpy(m)
            else:
                self.weights['m'] = m
            
            if isinstance(n, np.ndarray):
                self.weights['n'] = torch.from_numpy(n)
            else:
                self.weights['n'] = n
            
            # Add Who.weight (readout weights)
            who_keys = ['Who.weight', 'fc.weight', 'rnn.Who.weight']
            for key in who_keys:
                if key in weights_dict:
                    self.weights['Who.weight'] = weights_dict[key]
                    break
            
            # Add Wih.weight (input weights)
            wih_keys = ['Wih.weight', 'rnn.Wih.weight']
            for key in wih_keys:
                if key in weights_dict:
                    self.weights['Wih.weight'] = weights_dict[key]
                    break
            
            # Add Wih.bias if exists
            wih_bias_keys = ['Wih.bias', 'rnn.Wih.bias']
            for key in wih_bias_keys:
                if key in weights_dict:
                    self.weights['Wih.bias'] = weights_dict[key]
                    break
            
            # Create rnn_func with rank attribute
            rnn_func = type('obj', (object,), {})()
            rnn_func.rank = 2
            self.rnn_func = rnn_func
            self.units = units
    
    model_wrapper = SimpleModel(weights, m_tensor, n_tensor)
    rank_analyzer = RankAnalyzer(model_wrapper)
    
    return rank_analyzer, weights


def get_contours(U, V, Z, threshold=0.2):
    """
    Get contour paths for basin of attraction.
    """
    gen = contourpy.contour_generator(U, V, Z)
    contours = [Path(contour) for contour in gen.lines(threshold)]
    return contours


def identify_attractor_for_point(point, contours):
    """
    Identify which attractor (contour) a point belongs to.
    
    Args:
        point: Array of shape (2,) - kappa coordinates
        contours: List of Path objects representing attractor basins
        
    Returns:
        attractor_idx: Index of the attractor, or -1 if not found
    """
    point_tuple = tuple(point) if isinstance(point, (list, np.ndarray)) else point
    
    for i, contour_path in enumerate(contours):
        try:
            if contour_path.contains_point(point_tuple):
                return i
        except (TypeError, AttributeError):
            if hasattr(contour_path, 'contains_points'):
                point_array = np.array([point_tuple]).reshape(1, -1)
                if contour_path.contains_points(point_array)[0]:
                    return i
    
    return -1


def identify_task_type(x, data_generator, task_combination, training_mode="standard"):
    """
    Identify task type from input.
    """
    tasks_sorted = sorted(task_combination)
    n_tasks = len(tasks_sorted)
    
    if n_tasks == 2:
        channel_activities = [np.sum(np.abs(x[:, i]) > 1e-6) for i in range(n_tasks)]
        
        if training_mode == "parallel":
            vmin, vmax = data_generator.vmin, data_generator.vmax
            channel_binary_counts = []
            channel_continuous_counts = []
            
            for i in range(n_tasks):
                # Count binary values (exactly vmin or vmax)
                binary_count = np.sum(np.isclose(x[:, i], vmin) | np.isclose(x[:, i], vmax))
                channel_binary_counts.append(binary_count)
                
                # Count continuous values (not exactly vmin or vmax, but non-zero)
                non_binary_mask = ~(np.isclose(x[:, i], vmin) | np.isclose(x[:, i], vmax))
                continuous_count = np.sum(non_binary_mask & (np.abs(x[:, i]) > 1e-6))
                channel_continuous_counts.append(continuous_count)
            
            # For flipflop+line combination, use value type to distinguish
            if 'flipflop' in tasks_sorted and 'line' in tasks_sorted:
                flipflop_idx = tasks_sorted.index('flipflop')
                line_idx = tasks_sorted.index('line')
                
                # Flipflop uses binary values, line uses continuous values
                # Check which channel has more continuous (non-binary) values
                if channel_continuous_counts[line_idx] > channel_continuous_counts[flipflop_idx]:
                    return 'line', line_idx
                elif channel_binary_counts[flipflop_idx] > channel_binary_counts[line_idx]:
                    return 'flipflop', flipflop_idx
                # If ambiguous, use the channel with more total activity
                elif channel_activities[line_idx] > channel_activities[flipflop_idx]:
                    return 'line', line_idx
                else:
                    return 'flipflop', flipflop_idx
            
            # For other combinations, use original logic
            if 'flipflop' in tasks_sorted:
                flipflop_idx = tasks_sorted.index('flipflop')
                if channel_binary_counts[flipflop_idx] > max([c for i, c in enumerate(channel_binary_counts) if i != flipflop_idx]):
                    return 'flipflop', flipflop_idx
            
            if 'cycle' in tasks_sorted:
                cycle_idx = tasks_sorted.index('cycle')
                if channel_activities[cycle_idx] > 0:
                    return 'cycle', cycle_idx
            
            if 'line' in tasks_sorted:
                line_idx = tasks_sorted.index('line')
                if channel_activities[line_idx] > 0:
                    return 'line', line_idx
            
            dominant_idx = np.argmax(channel_activities)
            return tasks_sorted[dominant_idx], dominant_idx
        else:
            active_channels = [i for i in range(n_tasks) if channel_activities[i] > 0]
            
            if len(active_channels) == 1:
                task_idx = active_channels[0]
                return tasks_sorted[task_idx], task_idx
            elif len(active_channels) > 1:
                dominant_idx = active_channels[np.argmax([channel_activities[i] for i in active_channels])]
                return tasks_sorted[dominant_idx], dominant_idx
            else:
                return tasks_sorted[0], 0
    
    return tasks_sorted[0], 0


def run_trials_and_project_to_kappa(model, data_generator, rank_analyzer, task_combination, 
                                     n_trials_per_task=20, training_mode="standard", steps=200):
    """
    Run trials and project trajectories to kappa space.
    
    Returns:
        trajectories_kappa: Dict mapping task to list of kappa trajectories
        final_kappas: Dict mapping task to final kappa positions
        attractor_assignments: Dict mapping task to list of attractor indices
    """
    all_trajectories = {task: [] for task in task_combination}
    all_final_kappas = {task: [] for task in task_combination}
    task_counts = {task: 0 for task in task_combination}
    
    print(f"Running trials and projecting to kappa space...")
    
    if training_mode == "parallel":
        # For parallel mode: generate n_trials_per_task * 2 trials and randomly assign labels
        total_trials = n_trials_per_task * len(task_combination)
        print(f"  Parallel mode: generating {total_trials} trials and randomly assigning task labels...")
        
        # Generate all trials
        all_trial_data = []
        for i in range(total_trials):
            x, y = data_generator.generate_training_trial()
            x = x[np.newaxis, :, :]
            all_trial_data.append((x, y))
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{total_trials} trials...")
        
        # Randomly assign task labels
        np.random.seed(42)  # For reproducibility
        task_labels = []
        for task in task_combination:
            task_labels.extend([task] * n_trials_per_task)
        np.random.shuffle(task_labels)
        
        # Process trials with assigned labels
        for i, ((x, y), task_label) in enumerate(zip(all_trial_data, task_labels)):
            # Run model forward
            predictions = model.predict(x)
            states = predictions['state'][0]  # (time_steps, hidden_units)
            
            # Project to kappa space
            kappa_trajectory = np.vstack([rank_analyzer.state_to_kappa(s) for s in states])
            final_kappa = kappa_trajectory[-1]
            
            all_trajectories[task_label].append(kappa_trajectory)
            all_final_kappas[task_label].append(final_kappa)
            task_counts[task_label] += 1
            
            if (i + 1) % 10 == 0:
                progress_str = ", ".join([f"{task}={task_counts[task]}/{n_trials_per_task}" 
                                         for task in task_combination])
                print(f"  Progress: {progress_str}")
    else:
        # For standard/orthogonal mode: use identify_task_type
        max_attempts = n_trials_per_task * len(task_combination) * 3
        attempts = 0
        
        while any(count < n_trials_per_task for count in task_counts.values()) and attempts < max_attempts:
            attempts += 1
            
            x, y = data_generator.generate_training_trial()
            x = x[np.newaxis, :, :]
            
            task_label, task_num = identify_task_type(x[0], data_generator, task_combination, training_mode)
            
            if task_label in task_counts and task_counts[task_label] < n_trials_per_task:
                # Run model forward
                predictions = model.predict(x)
                states = predictions['state'][0]  # (time_steps, hidden_units)
                
                # Project to kappa space
                kappa_trajectory = np.vstack([rank_analyzer.state_to_kappa(s) for s in states])
                final_kappa = kappa_trajectory[-1]
                
                all_trajectories[task_label].append(kappa_trajectory)
                all_final_kappas[task_label].append(final_kappa)
                task_counts[task_label] += 1
                
                if sum(task_counts.values()) % 10 == 0:
                    progress_str = ", ".join([f"{task}={task_counts[task]}/{n_trials_per_task}" 
                                             for task in task_combination])
                    print(f"  Progress: {progress_str}")
    
    print(f"  Final: {task_counts}")
    
    return all_trajectories, all_final_kappas


def plot_kappa_trajectories_with_basins(rank_analyzer, trajectories_kappa, final_kappas, 
                                        task_combination, contours, U, V, Z, save_path="kappa_trajectories.png"):
    """
    Plot trajectories in kappa space with basin of attraction colored by attractor.
    """
    # Identify which attractor each final kappa belongs to
    attractor_assignments = {}
    points_outside_basins = {task: [] for task in task_combination}
    
    for task in task_combination:
        attractor_assignments[task] = []
        for traj_idx, final_kappa in enumerate(final_kappas[task]):
            attractor_idx = identify_attractor_for_point(final_kappa, contours)
            attractor_assignments[task].append(attractor_idx)
            
            # Check if point is outside all basins
            if attractor_idx == -1:
                points_outside_basins[task].append(traj_idx)
                # Calculate distance to nearest attractor (Z value)
                # Interpolate Z value at this point
                from scipy.interpolate import griddata
                z_value = griddata((U.flatten(), V.flatten()), Z.flatten(), 
                                  (final_kappa[0], final_kappa[1]), method='nearest')
                if z_value is not None:
                    print(f"  Warning: {task} trial {traj_idx} final point outside basins, Z={z_value:.4f}")
    
    # Print summary
    for task in task_combination:
        n_outside = len(points_outside_basins[task])
        n_total = len(final_kappas[task])
        if n_outside > 0:
            print(f"  {task.capitalize()}: {n_outside}/{n_total} points outside basins")
    
    # Get unique attractors and assign colors
    all_attractors = set()
    for task in task_combination:
        all_attractors.update(attractor_assignments[task])
    all_attractors = sorted([a for a in all_attractors if a >= 0])
    
    # Color map for attractors (low saturation, pastel colors)
    n_attractors = len(all_attractors)
    # Use Pastel1 colormap which has lower saturation than Set3
    base_colors = plt.cm.Pastel1(np.linspace(0, 1, max(n_attractors, 9)))
    # Further reduce saturation by mixing RGB channels with white (0.7 color + 0.3 white)
    attractor_colors = np.array([np.append(c[:3] * 0.7 + 0.3, c[3]) for c in base_colors])
    
    # Task colors (low saturation, pastel colors)
    task_colors = {
        'flipflop': '#7BAFD4',  # Soft blue (pastel blue)
        'line': '#F4A460',      # Soft orange (sandy brown)
        'cycle': '#90EE90'      # Soft green (light green)
    }
    
    # Create figure with square aspect ratio
    fig, ax = plt.subplots(figsize=(2.85, 2.85))
    
    # Plot basin contours
    for i, contour_path in enumerate(contours):
        if i < len(attractor_colors):
            color = attractor_colors[i]
        else:
            color = 'lightgrey'
        
        # Special handling for central attractor (if exists)
        if len(contours) == 9:
            vertices = contour_path.vertices
            if len(vertices) > 0:
                centroid_x = np.mean(vertices[:, 0])
                centroid_y = np.mean(vertices[:, 1])
                if np.abs(centroid_x) < 2 and np.abs(centroid_y) < 2:
                    color = 'lightgrey'
        
        patch = PathPatch(contour_path, color=color, alpha=0.3, linewidth=0.5, 
                         edgecolor='gray', zorder=0)
        ax.add_patch(patch)
    
    # Plot trajectories colored by task (consistent with legend)
    for task in task_combination:
        task_color = task_colors.get(task, 'black')
        marker = 'o' if task == 'flipflop' else 's' if task == 'line' else '^'
        
        for traj_idx, (trajectory, attractor_idx) in enumerate(zip(
            trajectories_kappa[task], attractor_assignments[task])):
            
            # Use task color for trajectory line
            traj_color = task_color
            
            # Plot trajectory with task color (make it more visible)
            # Only plot if trajectory has more than 1 point
            # if len(trajectory) > 1:
            #     ax.plot(trajectory[:, 0], trajectory[:, 1], 
            #            color=traj_color, linewidth=0.8, alpha=0.7, zorder=2, 
            #            label=None)  # Don't add to legend for each trajectory
            
            # Mark final point with task color and marker
            # Use black border for points that converged to a basin, no border for others
            final_kappa = trajectory[-1]
            if attractor_idx >= 0:
                # Converged to a basin: use black border
                edge_color = 'black'
                edge_width = 1.0
            else:
                # Not converged: no border (same color as fill)
                edge_color = traj_color
                edge_width = 0.0
            
            ax.scatter(final_kappa[0], final_kappa[1], 
                      color=traj_color, s=30, alpha=0.6, zorder=3, 
                      marker=marker, edgecolors=edge_color, linewidths=edge_width)
    
    # Set limits to ensure square aspect ratio (same range for both axes)
    max_k1 = rank_analyzer.max_kappa(0) - 2
    max_k2 = rank_analyzer.max_kappa(1) - 2
    # Use the maximum of both to ensure square display area
    max_range = max(max_k1, max_k2)
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    
    ax.set_xlabel(r'$\kappa_1$', fontsize=10, color='black')
    ax.set_ylabel(r'$\kappa_2$', fontsize=10, color='black')
    
    # Show axis ticks with black color
    ax.tick_params(axis='both', which='major', labelsize=8, colors='black')
    ax.tick_params(axis='both', which='minor', labelsize=6, colors='black')
    
    # Set axis spines (borders) to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)
    
    # # Add black axis lines at x=0 and y=0
    # ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-', zorder=1, alpha=0.5)
    # ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-', zorder=1, alpha=0.5)
    
    # Task name mapping for legend
    task_name_map = {
        'flipflop': 'Flip-Flop',
        'line': 'Line',
        'cycle': 'Limit Cycle'
    }
    
    # Create legend with both task markers and attractor colors
    legend_elements = []
    for task in task_combination:
        marker = 'o' if task == 'flipflop' else 's' if task == 'line' else '^'
        task_label = task_name_map.get(task, task.capitalize())
        legend_elements.append(plt.Line2D([0], [0], marker=marker, linestyle='-', 
                                          color=task_colors.get(task, 'black'), 
                                          linewidth=1, markersize=5, 
                                          label=task_label, 
                                          markerfacecolor=task_colors.get(task, 'black'),
                                          markeredgecolor='black', markeredgewidth=0.5))
    
    ax.legend(handles=legend_elements, fontsize=8, loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Display number of attractors in the top-left corner
    n_basins = len(contours)
    ax.text(0.02, 0.98, f'Number of Basins: {n_basins}', 
            transform=ax.transAxes, fontsize=9, color='black',
            verticalalignment='top', horizontalalignment='left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def main():
    """
    Main function to plot trajectories in kappa space with basins.
    """
    
    task_combination = ['line', 'cycle']  # Change this to switch task combinations
    training_mode = "parallel"  # Options: "standard", "orthogonal", "parallel"
    model_path = None  # Will search automatically if None
    
    units = 100
    n_bits = 2
    steps = 200  # Increased for better convergence to attractors
    n_trials_per_task = 100  # Number of example trials per task
    instance = 0
    n_points_basin = 121  # Grid points for basin calculation
    
    pytorchtools.device = torch.device("cpu")
    
    # Validate task combination
    valid_tasks = ['flipflop', 'line', 'cycle']
    if len(task_combination) != 2 or not all(task in valid_tasks for task in task_combination):
        print(f"Error: task_combination must be a list of 2 tasks from {valid_tasks}")
        return
    
    # Find model
    if model_path is None:
        print(f"Searching for rank 2 model with tasks: {task_combination}, mode: {training_mode}...")
        model_path = find_model_path(task_combination=task_combination, training_mode=training_mode, 
                                     rank=2, instance=instance)
        
        if model_path is None:
            print(f"\nError: Could not find a model for tasks {task_combination} with training mode '{training_mode}'.")
            models_dir = PathLib("models")
            if models_dir.exists():
                print("\nAvailable models:")
                for folder in sorted(models_dir.iterdir()):
                    if folder.is_dir() and "rank2" in folder.name:
                        print(f"  - {folder.name}")
            return
    
    print(f"\nLoading model from: {model_path}")
    print(f"Task combination: {task_combination}")
    print(f"Training mode: {training_mode}")
    
    # Load model and create rank analyzer
    rank_analyzer, weights = load_model_and_create_rank_analyzer(
        model_path, units=units, n_bits=n_bits, steps=steps
    )
    
    # Create data generator
    data_generator = get_data_generator(task_combination, training_mode, n_bits)
    data_generator._steps = steps
    
    # Create model architecture for running trials
    model = Rank2Architechture(
        units=units,
        inputs=data_generator.n_inputs,
        outputs=data_generator.n_outputs,
        recurrent_bias=False,
        readout_bias=True
    )
    model.set_model_dir(str(PathLib(model_path).parent))
    model.load_weights(weights)
    
    print("Model loaded successfully!")
    
    # Generate basin of attraction
    print(f"\nGenerating basin of attraction (grid: {n_points_basin}x{n_points_basin})...")
    U, V, Z, initial_states = rank_analyzer.kappa_UVZ(n_points_basin)
    contours = get_contours(U, V, Z, threshold=0.2)
    print(f"  Found {len(contours)} attractor basins")
    print(f"  Z value range: [{Z.min():.4f}, {Z.max():.4f}]")
    print(f"  Z value mean: {Z.mean():.4f}")
    
    # Run trials and project to kappa space
    trajectories_kappa, final_kappas = run_trials_and_project_to_kappa(
        model, data_generator, rank_analyzer, task_combination,
        n_trials_per_task=n_trials_per_task, training_mode=training_mode, steps=steps
    )
    
    # Plot
    task_str = '_'.join(sorted(task_combination))
    output_path = f"kappa_trajectories_{task_str}_{training_mode}.png"
    plot_kappa_trajectories_with_basins(
        rank_analyzer, trajectories_kappa, final_kappas, task_combination,
        contours, U, V, Z, save_path=output_path
    )


if __name__ == "__main__":
    main()

