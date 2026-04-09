from model.pt_models import Rank2Architechture
from data.custom_data_generator import *
from analysis.basinofattraction import BasinOfAttraction
from model.model_wrapper import ModelWrapper
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image
import tempfile
import shutil

kwargs = {'architecture_func': Rank2Architechture,
          'units': 100,
          'train_data': FlipFlopLineGenerator(n_bits=2),
          'instance_range': [100],
          'recurrent_bias': True,
          'readout_bias': True,
          'freeze_params': None,
          'weight_init_func': None}

full_wrapper = ModelWrapper(**kwargs)

# Configuration
CHECKPOINTS = range(10, 1000, 10)
SAVE_GIF = True
GIF_OUTPUT_PATH = None
GIF_DURATION = 50

def create_gif_from_checkpoints(wrapper, analyses, checkpoints, gif_path=None, duration=500):
    """
    Create a GIF animation from analysis checkpoints.
    
    Args:
        wrapper: ModelWrapper instance
        analyses: List of analysis classes
        checkpoints: List or range of checkpoint numbers
        gif_path: Output path for GIF (None for auto-generated)
        duration: Duration of each frame in milliseconds
    """
    from model.model_wrapper import InnerModelWrapper
    
    temp_dir = tempfile.mkdtemp()
    frame_paths = []
    
    try:
        print(f"Generating frames for GIF...")
        
        for inst in wrapper.instance_range:
            inner_wrapper = InnerModelWrapper(wrapper.architecture, wrapper.name, inst)
            
            checkpoints_dict = inner_wrapper.get_checkpoints_weights(checkpoints, valid=False)
            print(f"Found checkpoints: {sorted([k for k in checkpoints_dict.keys() if k != -1])}")
            
            x, y = wrapper.test_data.get_data()
            
            for chkpt in sorted(checkpoints_dict.keys()):
                if chkpt == -1:
                    continue
                    
                print(f"  Processing checkpoint {chkpt}...")
                inner_wrapper.architecture.load_weights(checkpoints_dict[chkpt])
                pred = inner_wrapper.architecture.predict(x)
                outputs, states = pred['output'], pred['state']
                
                for analysis_class in analyses:
                    analysis = analysis_class(
                        inner_wrapper.architecture,
                        wrapper.test_data,
                        wrapper.name,
                        inst,
                        outputs,
                        states,
                        chkpt
                    )
                    
                    original_save_plot = analysis.save_plot
                    
                    def temp_save_plot(plt_obj, desc='', format='png'):
                        frame_path = os.path.join(temp_dir, f'frame_chkpt{chkpt:04d}.png')
                        Path(temp_dir).mkdir(parents=True, exist_ok=True)
                        plt_obj.savefig(frame_path, bbox_inches='tight', dpi=150)
                        plt_obj.close()
                        if frame_path not in frame_paths:
                            frame_paths.append(frame_path)
                    
                    analysis.save_plot = temp_save_plot
                    
                    analysis.run()
                    
                    analysis.save_plot = original_save_plot
        
        if frame_paths:
            print(f"\nCreating GIF from {len(frame_paths)} frames...")
            
            images = []
            for frame_path in sorted(frame_paths):
                if os.path.exists(frame_path):
                    img = Image.open(frame_path)
                    images.append(img)
            
            if len(images) > 0:
                if gif_path is None:
                    model_name = wrapper.name
                    instance = list(wrapper.instance_range)[0]
                    gif_path = f'analysis_results/{model_name}_i{instance}_basin_attraction_animation.gif'
                
                Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
                
                images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=duration,
                    loop=0
                )
                print(f"✓ GIF saved to: {gif_path}")
            else:
                print("✗ No valid images to create GIF")
        else:
            print("✗ No frames generated")
            
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory")

if SAVE_GIF:
    create_gif_from_checkpoints(
        full_wrapper,
        [BasinOfAttraction],
        CHECKPOINTS,
        gif_path=GIF_OUTPUT_PATH,
        duration=GIF_DURATION
    )
else:
    full_wrapper.get_analysis_checkpoints([BasinOfAttraction], checkpoints=CHECKPOINTS)
