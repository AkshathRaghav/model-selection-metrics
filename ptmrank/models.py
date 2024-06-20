import json, os 
from typing import List, Union

class ExhaustedModelsError(Exception):
    pass

class Model: 
    def __init__(self, model_name: str, ckpt: str = None, embedding_function=None): 
        self.model_name = model_name 
        self.ckpt = ckpt 
        self.embedding_function = embedding_function    

class Models: 
    def __init__(self, models: Union[List[Model], List[str], None]): 
        if models is None:  
            raise ExhaustedModelsError("No models found for the task!")
        
        if isinstance(all(models), str):
            models = [Model(model_name) for model_name in models]
        
        assert all(isinstance(model, Model) for model in models), "`models` parameter needs either List[str] or List[Model]!"

        self.models = models 
        self.idx = 0

    def __len__(self):
        return len(self.models)
    
    def __call__(self):
        if self.idx < len(self.models):
            model = self.models[self.idx]
            self.idx += 1
            return model
        else:
            raise ExhaustedModelsError("No more models left!")

class ModelsPeatMOSS: 
    def __init__(self, mapping_file: str): 
        assert os.path.exists(mapping_file), f"Mapping file {mapping_file} does not exist!"
        
        self.tasks_all = [
            "Feature Extraction",
            "Text-to-Image",
            "Image-to-Text",
            "Image-to-Video",
            "Text-to-Video",
            "Visual Question Answering",
            "Document Question Answering",
            "Graph Machine Learning",
            "Text-to-3D",
            "Image-to-3D",
            "Depth Estimation",
            "Image Classification",
            "Object Detection",
            "Image Segmentation",
            "Image-to-Image",
            "Unconditional Image Generation",
            "Video Classification",
            "Zero-Shot Image Classification",
            "Mask Generation",
            "Zero-Shot Object Detection",
            "Text Classification",
            "Token Classification",
            "Table Question Answering",
            "Question Answering",
            "Zero-Shot Classification",
            "Translation",
            "Summarization",
            "Conversational",
            "Text Generation",
            "Text2Text Generation",
            "Fill-Mask",
            "Sentence Similarity",
            "Text-to-Speech",
            "Text-to-Audio",
            "Automatic Speech Recognition",
            "Audio-to-Audio",
            "Audio Classification",
            "Voice Activity Detection",
            "Tabular Classification",
            "Tabular Regression",
            "Reinforcement Learning",
            "Robotics"
        ]

        self.tasks_all = [t.lower().replace(' ', '-') for t in self.tasks_all]
        self.task_models = self.create_task_model_mapping(self.load_json_records(mapping_file)) 

    def __getitem__(self, task: str):
        if task in self.task_models:
            return Models(self.task_models[task])
        else:
            return Models([])
            
    def create_task_model_mapping(self, json_records, truncate=False):
        task_model_mapping = {}
        for task_one in self.tasks_all: 
            task_model_mapping[task_one] = [] 
        

        for record in json_records:
            tasks = record[0].split(',')
            model_identifier = record[1]  

            for task in tasks:
                if task in task_model_mapping:
                    if model_identifier not in task_model_mapping[task]:
                        task_model_mapping[task].append(model_identifier)

        if truncate: 
            task_models = {} 
            for task, models in task_model_mapping.items():
                if len(models) > 5:
                    task_models[task] = models
            return task_models

        return task_model_mapping
    
    def load_json_records(self, mapping_file: str):
        with open(mapping_file, 'r') as f: 
            records = json.load(f)
        return records

