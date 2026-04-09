from analysis.kappaplane import KappaPlane
from model.model_wrapper import ModelWrapper
from model.pt_models import Rank2Architechture
from data.data_generator import FamilyOfTasksGenerator
from data.functions import *
from data.custom_data_generator import FlipFlopGenerator, LinesGenerator
from analysis.tasks_pca import TasksPCA

# TASKS = [X4Rotate(), X2Rotate(), XReverse(), X(), X2(), X4()]
TASKS = [FlipFlopGenerator(n_bits=2), LinesGenerator(n_bits=2)]

kwargs = {'architecture_func':Rank2Architechture,
          'units': 100,
          'train_data':FamilyOfTasksGenerator(TASKS),
          'instance_range': range(10, 11),
}

full_wrapper = ModelWrapper(**kwargs)
full_wrapper.train_model()
full_wrapper.analyze([KappaPlane, TasksPCA])
