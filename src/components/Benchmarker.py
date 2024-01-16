from .Model import Models 
from .LogME import LogME
from .LEEP import LEEP
from .NCE import NCE
from .FeatureExtractors import TextFeatureExtractor, ImageFeatureExtractor, AudioFeatureExtractor
import os
from .Dataloaders import DataLoader
import numpy as np
import pandas as pd 
import json
from tqdm import tqdm
import warnings 

class Benchmarker:
    def __init__(self, models: Models, logme=True, ckpt=False, regression = False, auto_increment_if_failed = False, leep=False, nce=False, test = False):
        self.models = models
        self.checkpoint = ckpt
        self.auto_increment_if_failed = auto_increment_if_failed
        self.test = test
        if self.test: 
            print('----------Testing Benchmarker----------')

        if int(logme) + int(leep) + int(nce) != 1:
            raise ValueError("ERROR: Only one of logme, leep, or nce can be True") 
        
        if logme:
            self.benchmark_model = LogME(regression=regression) 
            self.regression = regression    
            print(f'SUCCESS: LogME initialized with{"" if regression else " no"} regression')
        elif leep: 
            self.benchmark_model = LEEP()
            print('SUCCESS: LEEP initalized')
        else: 
            self.benchmark_model = NCE()
            print('SUCCESS: NCE initalized')
        
        self.extractor_groups = {
            TextFeatureExtractor: [
                'text-classification',
                'token-classification',
                'question-answering',
                'zero-shot-classification',
                'translation',
                'summarization',
                'conversational',
                'text-generation',
                'text2text-generation',
                'fill-mask',
                'sentence-similarity', 
                'feature-extraction'
            ],
            ImageFeatureExtractor: [
                'image-classification',
                'object-detection',
                'image-segmentation', 
                'video-classification'
            ],
            "Multimodal Not Supported": [
                'text-to-image',
                'image-to-text', 
                'table-question-answering'
            ],
            AudioFeatureExtractor: [
                'automatic-speech-recognition',
                'audio-classification'
            ],
        }

    def __call__(self, design_specs):
        task = design_specs['task']
        dataloader = design_specs['dataset']
        if not isinstance(dataloader, DataLoader): 
            raise ValueError('Dataset should be a `DataLoader` class!!') 
        n = design_specs['n']
        return self.benchmark_models(task, dataloader, n)

    def benchmark_models(self, task, dataloader, n=5): 
        if self.checkpoint:
            checkpoint_pth = f'./checkpoints/{task}_{str(dataloader)}' 
            try: 
                os.mkdir(checkpoint_pth)
            except: 
                pass

        extractor_group = [x for x in self.extractor_groups if task in self.extractor_groups[x]][0]
        if type(extractor_group) == str: 
            raise ValueError(f'ERROR: {extractor_group}')
        
        print('TASK: ', task)

        models = self.models(task, n)
        
        print('MODELS: ', [model[0] for model in models])

        if len(models) == 0:
            raise Error('ERROR: No models selected!')
        
        benchmarks = {}
        
        check = False
        for model_name, base_model_identifier, _, _ in models: 
            
            if check and self.auto_increment_if_failed: 
                self.models(task, 1)
            
            check = False
            
            print('----------------------------------------')
            print('Model_name', model_name)
            if self.checkpoint:
                model_checkpoint_pth = checkpoint_pth + f"/{model_name.replace('/', '_')}/"
                try: 
                    os.mkdir(model_checkpoint_pth)
                except: 
                    pass 
            
            use_checkpoint = False 
            if os.path.exists(model_checkpoint_pth+'/config.json'): 
                try: 
                    config = json.load(model_checkpoint_pth+'/config.json')
                    use_checkpoint=True
                except: 
                    use_checkpoint=False

            if not self.test: 
                if not use_checkpoint:
                    extractor = extractor_group(task, model_name)
                    if not extractor.feature_extractor or not extractor.model:
                        print(f'     ERROR: Loading {model_name}, aborting benchmark!')
                        check = True 
                        continue 
                    else: 
                        pass
                else: 
                    pass
            print('     SUCCESS: Extractor Loaded')
            if not self.test: 
                if not use_checkpoint: 
                    try: 
                        features, targets = self.extract_features_and_targets(extractor, dataloader, n)
                    except ValueError as e: 
                        print(e)
                        check = True 
                        continue    
                else: 
                    features, targets = config['features_pth'], config['targets_pth']

            print('     SUCCESS: Feature Extraction Complete.')
            print('     NOTE: Starting fitting the benchmarker.')
            if not self.test: 
                score = self.fit(features, targets)
            else: 
                score = 1
            print(f'     SUCCESS: Benchmarker completed with score: {score}.')
            print('----------------------------------------')

            benchmarks[model_name] = score
            
            if self.checkpoint:
                np.save(model_checkpoint_pth+'/features.npy', features)
                np.save(model_checkpoint_pth+'/targets.npy', targets)
                config = { 
                    'model_name': model_name,
                    'dataset_name': str(dataloader),
                    'score': score,
                    'features_pth': model_checkpoint_pth+'/features.npy',
                    'targets_pth': model_checkpoint_pth+'/targets.npy'
                }
                with open(model_checkpoint_pth+'/config.json', 'w') as fp:
                    json.dump(config, fp)
            del use_checkpoint, features, targets
        return benchmarks
    
    def extract_features_and_targets(self, extractor, dataloader, n):
        print('     NOTE: Extracting Features and Targets')
        dataloader.reset()
        features_list = []
        labels_list = []

        progress_bar = tqdm(total=dataloader.max_idx, desc='       Progress: ')

        batch = dataloader() 
        while batch: 
            features, labels = extractor(batch)
            
            if not len(features) and not len(labels): 
                batch = dataloader()
                continue

            features_list.append(features)
            if type(labels) != np.ndarray: 
                labels = np.array([labels])
            labels_list.append(labels)

            progress_bar.update(1)

            batch = dataloader()

        progress_bar.close()

        if len(features_list) < n//2: 
            raise ValueError('ERROR: Not enough features extracted! (Could be sampling_rate error for AudioModels).')

        f = np.concatenate(features_list, axis=0)
        y = np.concatenate([np.array(self.convert_labels_with_pandas(labels_list))], axis=0)

        f = f.reshape(f.shape[0], -1)

        if not self.regression:
            y = y.astype(int)
        else:
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

        print('     NOTE: Features Shape: ', f.shape)
        print('     NOTE: Labels Shape: ', y.shape)

        return f, y
    
    def convert_labels_with_pandas(self, labels_list):
        flat_labels = [label for sublist in labels_list for label in sublist]
        labels_series = pd.Series(flat_labels)
        labels_series = labels_series.astype('category').cat.codes
        return labels_series.values
            
    def fit(self, features, labels):
        warnings.filterwarnings("ignore")
        self.benchmark_model.reset()
        return self.benchmark_model.fit(features, labels)
    
    def is_regression(self): 
        if self.test: 
            print('Testing is_regression')
            return False

        if all(isinstance(label, (int, float)) for label in self.labels):
            unique_labels = set(labels)
            if len(unique_labels) > len(labels) * 0.1:  
                return True

            if all(isinstance(label, int) for label in labels):
                return False

        return False