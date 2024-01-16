from PIL import Image
import pandas as pd
import numpy as np
import os
from PIL import Image
import pydub

class DataLoader: 
    def __init__(self): 
        pass 

class ImageDataLoader(DataLoader): 
    def __init__(self, inputs: list[list[str, str]], name, shuffle=False, n=None):
        self.name=name
        self.img_paths = [x[0] for x in inputs] 
        self.targets = [x[1] for x in inputs]
        self.idx = 0
        self.max_idx = n
        self.len = len(self.targets)
        self.shuffle = shuffle  
        self.dataframe = self.convert_to_pd()

    def __str__(self): 
        return self.name
        
    def __len__(self):
        return self.len  
    
    def convert_to_pd(self): 
        df = pd.DataFrame({'img_path': self.img_paths, 'target': self.targets})
        if self.shuffle: 
            df = df.sample(frac=1).reset_index(drop=True)
        return df
    
    def __call__(self): 
        self.idx += 1
        if (self.idx >= self.len) or (self.max_idx and self.idx > self.max_idx):
            print('     Dataloader exhausted!') 
            return None
        row = self.dataframe.iloc[self.idx]
        image = None
        with Image.open(row['img_path']) as f: 
            image = np.array(f)
        return (image, row['target'])
    
    def reset(self): 
        self.idx = 0   

class TextDataLoader(DataLoader): 
    def __init__(self, inputs: list[list[str, str]], name, shuffle=False, n=None):
        self.name=name
        self.texts = [x[0] for x in inputs] 
        self.targets = [x[1] for x in inputs]
        self.idx = 0
        self.max_idx = n
        self.len = len(self.targets)
        self.shuffle = shuffle  
        self.dataframe = self.convert_to_pd()
    
    def __str__(self): 
        return self.name
        
    def __len__(self):
        return self.len  
    
    def convert_to_pd(self): 
        df = pd.DataFrame({'input': self.texts, 'target': self.targets})
        if self.shuffle: 
            df = df.sample(frac=1).reset_index(drop=True)
        return df
    
    def __call__(self): 
        self.idx += 1
        if (self.idx >= self.len) or (self.max_idx and self.idx > self.max_idx):
            print('     Dataloader exhausted!') 
            return None
        row = self.dataframe.iloc[self.idx]
        return (row['input'], row['target'])
    
    def reset(self): 
        self.idx = 0   

class AudioDataLoader(DataLoader):
    def __init__(self, inputs: list[list[str, str]], name, type: str, normalized:bool, shuffle=False, n=None):
        self.name=name
        self.audio_paths = [x[0] for x in inputs] 
        self.targets = [x[1] for x in inputs]
        self.type=type
        self.normalized=normalized
        self.idx = 0
        self.max_idx = n
        self.len = len(self.targets)
        self.shuffle = shuffle  
        self.dataframe = self.convert_to_pd()

    def __str__(self): 
        return self.name
        
    def __len__(self):
        return self.len  
    
    def convert_to_pd(self): 
        df = pd.DataFrame({'audio_path': self.audio_paths, 'target': self.targets})
        if self.shuffle: 
            df = df.sample(frac=1).reset_index(drop=True)
        return df
    
    def __call__(self): 
        self.idx += 1
        if (self.idx >= self.len) or (self.max_idx and self.idx > self.max_idx):
            print('     Dataloader exhausted!') 
            return None
        row = self.dataframe.iloc[self.idx]
        return ((self.read(row['audio_path'], type=self.type, normalized=self.normalized)), row['target'])
    
    def reset(self): 
        self.idx = 0
    
    def read(self, filename, type='wav', normalized=False):
      with open(filename, 'rb') as file1:
          a = pydub.AudioSegment.from_file(file1, format=type)
          y = np.array(a.get_array_of_samples())
          if a.channels == 2:
              y = y.reshape((-1, 2))
          if normalized:
              return a.frame_rate, np.float32(y) / 2**15
          else:
              return a.frame_rate, np.float32(y)