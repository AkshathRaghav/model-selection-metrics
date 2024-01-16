import logging 

def setup(config):
  checkpoint_pth = f"../checkpoints/{config['task']}_{str(config['dataset'])}"
  try:
      os.mkdir(checkpoint_pth)
  except:
      pass

  logging.getLogger().setLevel(logging.INFO)
  logging.basicConfig(filename=f'{checkpoint_pth}/log.log',encoding='utf-8',level=logging.DEBUG, filemode = 'w', format='%(process)d-%(levelname)s-%(message)s') 
  