import tensorflow as tf
import os
import sys
import time
import json
import random
import math
import numpy as np

# pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# visualisation
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import PIL

from mrcnn import visualize
from mrcnn import model as modellib, utils
from mrcnn.config import Config


from Config_Visage_Detection import Config_Visage_Detection

#Preparation des chemins
PATH = "../"
SAVE_MODEL_PATH ="poids.h5"
LOGS_DIRECTORY = os.path.join(PATH, "logs")
MODEL_DIR = os.path.join(PATH, "logs")



config = Config_Visage_Detection()

# visualisation de la configuration.
# config.display()

# INITIALISATION EN MODE ENTRAINEMENT
model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)
model_path = SAVE_MODEL_PATH

# chargement des poids du modèle déja entraîné
model.load_weights(model_path, by_name=True)