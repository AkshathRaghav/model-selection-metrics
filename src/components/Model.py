import json 

class Models: 
    def __init__(self): 
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
        self.task_models = self.create_task_model_mapping(self.load_json_records()) 
        self.idx = 0 

    def __call__(self, task, n):
        if task in self.task_models:
            vals = self.task_models[task][self.idx:self.idx+n]
            self.idx = n + 1
            return vals 
        else:
            return []
            
    def create_task_model_mapping(self, json_records, truncate=False):
        task_model_mapping = {}
        checks = {} 
        for task_one in self.tasks_all: 
            task_model_mapping[task_one] = [] 
            checks[task_one] = []
        

        for record in json_records:
            tasks = record[0].split(',')
            model_identifier = record[1]  
            base_model_identifier = record[2]
            dataset = record[4]
            metric = record[5]

            for task in tasks:
                if task in task_model_mapping:
                    if model_identifier not in task_model_mapping[task]:
                        if (model_identifier, base_model_identifier) not in checks[task]:
                            checks[task].append((model_identifier, base_model_identifier))
                            task_model_mapping[task].append([model_identifier, base_model_identifier, dataset, metric])

        if truncate: 
            task_models = {} 
            for task, models in task_model_mapping.items():
                if len(models) > 5:
                    task_models[task] = models
            return task_models

        return task_model_mapping
    
    def load_json_records(self):
        with open('../mapping.json') as f: 
            records = json.load(f)
        return records

