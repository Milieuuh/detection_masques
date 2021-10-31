# mrcnn
from mrcnn.config import Config

class Config_Visage_Detection(Config):
    # Nom du modèle
    NAME = "Visage_Detection"

    # Paramètres définisssant la taille du lot (batch size) égale à IMAGES_PER_GPU * GPU_COUNT
    IMAGES_PER_GPU = 5
    GPU_COUNT = 1

    # Nombre de catégories incluant le background
    NUM_CLASSES = 4  # 3 catégories (avec masque, sans masque, mal porté) + 1 background

    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50
    IMAGE_MAX_DIM = 256
    IMAGE_MIN_DIM = 256


