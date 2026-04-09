import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from analysis.analyzer import TaskFamilyAnalyzer
from analysis.task_family_aux import extract_lists_from_trials

plt.rcParams['axes.facecolor'] = 'white'
sns.set()


class FFPCA(TaskFamilyAnalyzer):
    """
    PCA analyzer specifically for flip-flop tasks.
    Extends TaskFamilyAnalyzer to handle flip-flop tasks which are not Function1D/Function2D.
    """
    name = 'ff_pca'

    def __init__(self, model, data_params, model_name, inst, outputs, states, checkpoints=None):
        # Call parent init but handle the case where tasks are not Function1D/Function2D
        try:
            super().__init__(model, data_params, model_name, inst, outputs, states, checkpoints)
        except (AttributeError, KeyError, TypeError) as e:
            # If parent init fails due to flip-flop task structure, 
            # initialize basic attributes manually
            self.model = model
            self.X, self.Y = data_params.get_data()
            self.data_params = data_params
            self.model_name = model_name
            self.instance = inst
            self.checkpoint = checkpoints
            self.outputs, self.states = outputs, states
            
            # Handle both single task generator and multi-task generator
            if hasattr(data_params, 'tasks'):
                # Multi-task generator (FamilyOfTasksGenerator)
                self.n_tasks = len(data_params.tasks)
                self.TASKS = np.copy(data_params.tasks)
            else:
                # Single task generator (e.g., FlipFlopGenerator)
                # Treat the generator itself as the task
                self.n_tasks = 1
                self.TASKS = [data_params]
            
            X = self.extract_input()
            self.points_dict, self.values_dict = {}, {}
            self.all_points_dict = {}
            for i, task in enumerate(self.TASKS):
                try:
                    max_delay = getattr(self.data_params, 'max_delay', 40)
                    D = extract_lists_from_trials(X, self.states[i::self.n_tasks], task, max_delay)
                    self.points_dict[task] = D['points']
                    self.values_dict[task] = D['output']
                    self.all_points_dict[task] = D['all_points']
                    if 'xn' in D:
                        self.xn = D['xn']
                    if 'xnm1' in D:
                        self.xnm1 = D['xnm1']
                except Exception:
                    # If extraction fails, use all states for this task
                    task_states = self.states[i::self.n_tasks]
                    if task_states.ndim > 2:
                        self.points_dict[task] = task_states.reshape(-1, task_states.shape[-1])
                    else:
                        self.points_dict[task] = task_states
                    self.values_dict[task] = np.zeros(len(self.points_dict[task]))
            
            self.all_points = np.vstack(list(self.points_dict.values()))
            self.task_dimensionality = None
    
    def extract_input(self):
        """Extract input data based on input type."""
        # Handle both single task generator and multi-task generator
        if hasattr(self.data_params, 'tasks'):
            n_tasks = len(self.data_params.tasks)
            if hasattr(self.data_params, 'input_type') and self.data_params.input_type == 'multi':
                X = self.X[::n_tasks, :, 0]
            else:
                X = self.X[::n_tasks, :, -1]
        else:
            # Single task generator - use all data
            if hasattr(self.data_params, 'input_type') and self.data_params.input_type == 'multi':
                X = self.X[:, :, 0]
            else:
                X = self.X[:, :, -1]
        return X

    def run(self):
        """
        Run PCA analysis on flip-flop task representations.
        """
        pca = PCA(2)
        pca.fit(self.all_points)
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(projection=None)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xticks([])
        ax.set_yticks([])
        
        markers = ['o', 'v', 'x', 's', 'p', '.']
        for task_num, task in enumerate(self.TASKS):
            # Get points for this task
            if task in self.points_dict:
                xn = self.points_dict[task]
            else:
                # Fallback: use all points if task-specific points not available
                xn = self.all_points
            
            # Get colors based on output values if available
            if task in self.values_dict:
                colors = self.values_dict[task]
                # Normalize colors for visualization
                if isinstance(colors, np.ndarray) and colors.ndim > 0:
                    colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-8)
            else:
                colors = 'blue'
            
            # Transform to 2D PCA space
            xn_2d = pca.transform(xn)
            ax.scatter(*xn_2d.transpose(), c=colors, marker=markers[task_num % len(markers)], 
                      alpha=0.6, s=20)
        
        self.save_plot(plt, 'pca', format='pdf')

