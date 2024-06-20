# PTMRank 
This repository acts as a package to assess the transferability of huggingface and pytorch hub models on popular metrics from 2021-2023. These metrics have been optimized for multi-core CPU and GPU acceleration. 

Say you have a dataset of source and target data. You aren't sure which PTM to use. Just search up some models you think might be useful. Choose one (or multiple) metrics below (find explanations for each of them below). Boom, you have a reliable ranking which tells you how transferrable a model is to the task you have at hand. Either the PTM can out of box do your task, or it can be fine-tuned for it.  

Original implementations of GBC, WDJE, EMMS, PED. TransRate was adapted from the original implementations. SDFA, PARC, LogME, LEEP and NCE were adapted from SDFA. 

# Roadmap 
- Current assumption -> CPU only 
- GBC and HScore implementation going on  


# Supported Metrics 

- LogME 
- NLEEP 
- Gaussian Bhattacharyya Coefficient (GBC)
- SDFA - https://github.com/TencentARC/SFDA/blob/main/metrics.py
- TransRate - https://proceedings.mlr.press/v162/huang22d/huang22d.pdf - https://github.com/Long-Kai/TransRate/blob/master/generate_transrate/transrate.py
- WDJE
- EMMS - https://arxiv.org/pdf/2308.06262
- Potential Energy Decline - https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Exploring_Model_Transferability_through_the_Lens_of_Potential_Energy_ICCV_2023_paper.pdf
- Addition of Regression - https://openaccess.thecvf.com/content/ICCV2023/papers/Gholami_ETran_Energy-Based_Transferability_Estimation_ICCV_2023_paper.pdf
- NCE 
- LEEP  
- H-Score 

Check out Tsinghua University's [TLib library](https://github.com/thuml/Transfer-Learning-Library) for an amazing implementation of other transfer learning methodologies, including alignment, reweighting, translation, and more.

```
from ptmrank.benchmarker import Benchmarker
from ptmrank.models import ModelsPeatMOSS
from Examples import (
    load_image_example,
    load_text_example,
    load_audio_example,
)

models = ModelsPeatMOSS('/home/aksha/Workbench/Research/Labs/duality/model_selection/metrics/model-selection-metrics/mapping.json')['image-classification']


# If you already have embeddings generated, attach them to your model name in the Models class.

benchmarker = Benchmarker(
    models, store_features='/home/aksha/Workbench/Research/Labs/duality/model_selection/metrics/model-selection-metrics/checkpoints', 
    logme=True, regression=False, auto_increment_if_failed=True
)

benchmarker('image-classification', load_image_example(), 5)
```


# Gaussian Bhattacharyya Coefficient (GBC)


| Artifacts | Links | 
| ----------| -----| 
| PDF | https://arxiv.org/pdf/2111.12780 | 
| Implementation | N/A (as of June 2024) | 


Simplified: Furthur away the class distributions are in the feature embedding space, the better 

GBC takes in the input sources after embedding it for feature extraction, and then calculates the target-class-specific mean and covariances, which it then uses to calculate the BC. The distance measures the overlap between class distributions in the embedding space, reflecting the potential transferability of the model to related tasks.

Note: We want the feature vectors at the start of the layers (before the task-specific ones) -> output_hidden_states[0]. For models without an initial embedding layer, the vector post-feature-extractor is take. 


  $$ 
  \mu_i = \frac{1}{N_i} \sum_{j=1}^{N_i} E_{ij}
  $$
  where $( mu_i )$ is the mean vector for class $( i )$, $( E_{ij} )$ is the $( j )$-th embedding in class $( i )$, and $( N_i )$ is the number of embeddings in class $( i )$.
  $$ 
  \Sigma_i = \frac{1}{N_i - 1} \sum_{j=1}^{N_i} (E_{ij} - \mu_i)(E_{ij} - \mu_i)^T
  $$ 
  where $( \Sigma_i )$ is the covariance matrix for class $( i )$. This formula assumes that $( E_{ij} )$ and $( \mu_i )$ are column vectors; $( (E_{ij} - \mu_i))$ is a column vector of deviations from the mean.
  $$ 
  D_B (c_i, c_j) = \frac{1}{8} (\mu_{c_i} - \mu_{c_j})^T (\sum(\mu_{c_i} - \mu_{c_j}))^{-1} + \frac{1}{2} \ln (\frac{|\frac{1}{2}(\sum_{c_i} + \sum_{c_j})|}{\sqrt{|\sum_{c_i}||\sum_{c_j}|}})
  $$ 
  $$
  BC(.,.) = exp - D_B(.,.)
  $$
  where $D_B$ is the Bhattacharyya distance and $BC$ is the Bhattacharyya coefficient. 
  $$ 
  GBC_{s \rightarrow t} = - \sum_{i, j} BC(c_i, c_j)
  $$  
  where $GBC$ returns the negative sum as the final transferability score. (Negative because higher BC means more overlap)

### Practical Considerations: 

- Dimensionality is a concern. $D_B$ expects embeddings in the (n_features, n_features) shape. For most models, n_features is quite large, and thus the covariance matrix and inverse matrix computations are very expensive. So, we apply PCA into a fixed space of 64 dimensions. 
- While estimating per-class covariance matrices, higher $N_c$ values cause extremely large covariance matrices, and lower $N_c$ values cause singular covariance matrices. So we use only the diagonal covariance matix.
  - Since we deal with the diagonal matrices, we are able to make some optimizations as well. 
    - We can calculate the inverse of the covariance matrix by taking the inverse of the diagonal elements only. 
    - We can calculate the determinant of the covariance matrix by taking the product of the diagonal elements only.


  
  