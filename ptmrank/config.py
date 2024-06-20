from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Union, List
from .models import Models 
from .metrics.Metric import Metric

class Config(BaseModel): 
    model_config = ConfigDict(arbitrary_types_allowed=True)
    task_name: str 
    models: Models
    metrics: Union[List[Metric], Metric, None] = None
    gpu: str = Field(
        default="-1", 
        description="GPU to use. Default is -1, which means CPU. If multiple GPUs are available, specify the GPU index to use."
    )
    extract_features_only: bool = Field(
        default=False, 
        description="If True, only extract features from the model and store them. Does not fit to metrics."
    )
    auto_increment_if_failed: bool = Field( 
        default=False, 
        description="If True, increment the model index if the task failed to run for current model"
    ) 
    persist_dir: str = Field(
        default="../ptmrankings", 
        description="Path to store embeddings and results"
    ) 
    



