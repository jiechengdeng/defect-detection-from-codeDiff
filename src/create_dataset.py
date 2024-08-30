import os
import sys

PROJECT_NAME = "model_evaluation_on_global_defects"
if not os.getcwd().endswith(PROJECT_NAME):
    os.chdir(os.path.join(os.getcwd()[:os.getcwd().find(PROJECT_NAME)], PROJECT_NAME))
sys.path.append(os.getcwd())
from common_library.dataset_util import DatasetUtil

DatasetUtil.create_codeXglue_dataset(mode='train')
