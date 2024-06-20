from ptmrank.benchmarker import Benchmarker
from ptmrank.models import ModelsPeatMOSS
from Examples import (
    load_image_example,
    load_text_example,
    load_audio_example,
)

models = ModelsPeatMOSS('/home/aksha/Workbench/Research/Labs/duality/model_selection/metrics/model-selection-metrics/mapping.json')['image-classification']
benchmarker = Benchmarker(
    models, store_features='/home/aksha/Workbench/Research/Labs/duality/model_selection/metrics/model-selection-metrics/checkpoints', 
    logme=True, regression=False, auto_increment_if_failed=True
)
benchmarker('image-classification', load_image_example(), 5)
