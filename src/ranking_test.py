from components.Dataloaders import ImageDataLoader, TextDataLoader, AudioDataLoader
from components.Benchmarker import Benchmarker
from components.Model import Models
from components.Examples import (
    load_image_example,
    load_text_example,
    load_audio_example,
)
import components.Logger as log
import logging

dataloader = load_text_example()

models = Models()

config = {"task": "text-classification", "dataset": dataloader, "n": 3}

log.setup(config)

benchmarker = Benchmarker(
    models, ckpt=True, logme=True, regression=False, auto_increment_if_failed=True
)

logging.critical(benchmarker(config))
