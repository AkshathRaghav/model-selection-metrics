from .models import Models, Model 
from .config import Config 
from .tools.feature_extractors import ImageEmbedder
from .metrics.GBC import GBC
import pickle 
import numpy as np 

def convert_to_rgb(data: np.ndarray) -> np.ndarray:
    return np.stack([data[0:1024].reshape(32, 32), data[1024:2048].reshape(32, 32), data[2048:3072].reshape(32, 32)], axis=-1)

with open(r"D:\model-selection-metrics\assets\cifar_10\data_batch_1", 'rb') as fo:
    data = pickle.load(fo, encoding='bytes')

labels = data[b'labels']
images = data[b'data']
print(images.shape)
del data 

models = Models( 
    models=["google/mobilenet_v2_1.0_224", "microsoft/resnet-50", "Falconsai/nsfw_image_detection", "aalonso-developer/vit-base-patch16-224-in21k-clothing-classifier"]
)

config = Config( 
    task_name="image-classification",
    models=models, 
    metrics=[GBC()], 
    gpu="0", 
    extract_features_only=True, 
    auto_increment_if_failed=False, 
    persist_dir="D:\model-selection-metrics\checkpoints"
)

for model in config.models.models: 
    extractor = ImageEmbedder(config.task_name, model)
    batch = np.zeros((10, 32, 32, 3), dtype=np.float32)
    for n, image in enumerate(images[:10]): 
        batch[n] = convert_to_rgb(image)
    tensor = extractor(batch, batch_size=5)
    print(tensor.shape)