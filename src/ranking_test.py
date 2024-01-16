import pandas as pd 
import numpy as np 
from components.Dataloaders import ImageDataLoader, TextDataLoader, AudioDataLoader
from components.Benchmarker import Benchmarker
from components.Model import Models

def load_aircraft_targets(): 
  with open("../assets/aircraft/images_family_train.txt") as file: 
    file_list = file.read().splitlines() 
  return [[f'../assets/aircraft/images/{file.split()[0]}.jpg', "-".join(file.split()[1:])] for file in file_list]

def load_tweet_classifier(): 
  df = pd.read_csv('./assets/tweet/Corona_NLP_train.csv', encoding='latin-1')
  df = df[['OriginalTweet', 'Sentiment']]
  return df.astype(str).values.tolist()

def load_esc50_classifier(): 
  df = pd.read_csv('./assets/esc_50/esc50.csv', encoding='latin-1')
  filenames, targets = df['filename'].tolist(), df['target'].tolist()
  return [[f'../assets/esc_50/audio/{filename}', target] for filename, target in zip(filenames, targets)]  
  
dataloader = ImageDataLoader(load_aircraft_targets(), name='airplane', shuffle=True, n=1000)
# dataloader = TextDataLoader(load_tweet_classifier(), name='tweets', shuffle=True, n=1000)
# dataloader = AudioDataLoader(load_esc50_classifier(), name='esc_50', type='wav', normalized=False, shuffle=True, n=1000)


models = Models()

config = { 
    'task': 'image-classification',
    'dataset': dataloader,  
    'n': 3
}

benchmarker = Benchmarker(models, ckpt=True, logme=True, regression=True, auto_increment_if_failed=True)

print(benchmarker(config))