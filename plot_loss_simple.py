"""
Simple script to plot training loss from existing loss_history.pkl files.

Usage:
    Directly specify the model directory paths for stage0 and stage1,
    then run this script in IDE.
"""

from plot_continual_learning_loss import plot_from_model_paths
import os

STAGE0_MODEL_PATH = "models/flipflop2bits_200_rank2_100"
STAGE1_MODEL_PATH = "models/cycles2bits_200_rank2_100"
INSTANCE = 1211
OUTPUT_FILE = None

if __name__ == '__main__':
    model_paths = [STAGE0_MODEL_PATH, STAGE1_MODEL_PATH]
    
    def extract_task_name(path):
        dir_name = os.path.basename(path)
        task_name = dir_name.split('_')[0]
        return task_name
    
    task_names = [extract_task_name(STAGE0_MODEL_PATH), extract_task_name(STAGE1_MODEL_PATH)]
    
    task_names_list = [task_names, task_names]
    
    print("Checking for loss history files...")
    for i, model_path in enumerate(model_paths):
        pkl_path = f"{model_path}/i{INSTANCE}/loss_history.pkl"
        if os.path.exists(pkl_path):
            print(f"  ✓ Stage {i}: Found {pkl_path}")
        else:
            print(f"  ✗ Stage {i}: Not found {pkl_path}")
            print(f"     Please check the path and instance number.")
    
    if OUTPUT_FILE is None:
        output_file = f'continual_learning_loss_instance_{INSTANCE}.png'
    else:
        output_file = OUTPUT_FILE
    
    print(f"\nPlotting training loss curves...")
    print(f"  Stage 0: {task_names[0]} -> {STAGE0_MODEL_PATH}")
    print(f"  Stage 1: {task_names[1]} -> {STAGE1_MODEL_PATH}")
    print(f"  Instance: {INSTANCE}")
    print(f"  Output: {output_file}\n")
    
    plot_from_model_paths(
        model_paths=model_paths,
        task_names=task_names_list,
        instance=INSTANCE,
        save_path=output_file
    )
