# LogME
This is the codebase for the following two papers:

- [LogME: Practical Assessment of Pre-trained Models for Transfer Learning](http://proceedings.mlr.press/v139/you21b.html), ICML 2021

- [Ranking and Tuning Pre-trained Models: A New Paradigm for Exploiting Model Hubs](https://arxiv.org/abs/2110.10545), JMLR 2022

**Note**: the second paper is an extended version of the first conference paper.

# How to use

```py 
from components.Dataloaders import ImageDataLoader, TextDataLoader, AudioDataLoader
from components.Benchmarker import Benchmarker
from components.Model import Models
from components.Examples import (
    load_image_example,
    load_text_example,
    load_audio_example,
)
import components.Logger as logger
import logging

dataloader = load_image_example()

models = Models()

config = {"task": "image-classification", "dataset": dataloader, "n": 3}

logger.setup(config) 

benchmarker = Benchmarker(
    models, ckpt=True, logme=True, regression=False, auto_increment_if_failed=True
)

logging.critical(benchmarker(config))
```

```py 
CRITICAL:root:LogME initialized with no regression
INFO:root:TASK: image-classification
INFO:root:----------------------------------------
INFO:root:Model_name: AkshatSurolia/ViT-FaceMask-Finetuned
Could not find image processor class in the image processor config or the model config. Loading based on pattern matching with the model's feature extractor configuration.
CRITICAL:root:Extractor Loaded
INFO:root:Extracting Features and Targets
       Progress: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [01:31<00:00,  6.01it/s]INFO:root:Dataloader exhausted!
CRITICAL:root:Feature Extraction Complete.
INFO:root:Starting fitting the benchmarker.
       PROGRESS: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 70/70 [00:00<00:00, 7497.29it/s]
CRITICAL:root:AkshatSurolia/ViT-FaceMask-Finetuned score: 0.8343365849503543.
INFO:root:----------------------------------------
INFO:root:----------------------------------------
INFO:root:Model_name: FredZhang7/google-safesearch-mini-v2
ERROR:root:Could not load FredZhang7/google-safesearch-mini-v2, aborting benchmark!
INFO:root:----------------------------------------
INFO:root:Model_name: IDEA-CCNL/Taiyi-vit-87M-D
CRITICAL:root:Extractor Loaded
INFO:root:Extracting Features and Targets
       Progress: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [01:27<00:00,  5.87it/s]INFO:root:Dataloader exhausted!
CRITICAL:root:Feature Extraction Complete.
INFO:root:Starting fitting the benchmarker.
       PROGRESS: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 70/70 [00:00<00:00, 784.61it/s]
CRITICAL:root:IDEA-CCNL/Taiyi-vit-87M-D score: 6.571173791054749.
INFO:root:----------------------------------------
INFO:root:----------------------------------------
INFO:root:Model_name: Intel/vit-base-patch16-224-int8-static
Could not find image processor class in the image processor config or the model config. Loading based on pattern matching with the model's feature extractor configuration.
ERROR:root:Could not load Intel/vit-base-patch16-224-int8-static, aborting benchmark!
CRITICAL:root:{'AkshatSurolia/ViT-FaceMask-Finetuned': 0.8343365849503543, 'IDEA-CCNL/Taiyi-vit-87M-D': 6.571173791054749}
```