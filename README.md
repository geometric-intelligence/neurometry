# Neural Geometry #

Official PyTorch implementation of the paper "Quantifying Local Extrinsic Curvature in Neural Manifolds" (2022).

[Francisco Acosta](https://web.physics.ucsb.edu/~facosta/), [Sophia Sanborn](https://www.sophiasanborn.com/), [Khanh Dao Duc](https://kdaoduc.com/), [Manu Mahdav](https://www.manusmad.com/) and [Nina Miolane](https://www.ninamiolane.com/).

The neural manifold hypothesis postulates that the activity of a neural population forms a low-dimensional manifold within the larger neural state space, whose structure reflects the structure of the encoded task variables. Many dimensionality reduction techniques have been used to study the structure of neural manifolds, but these methods do not provide an explicit parameterization of the manifold, and may not capture the global structure of topologically nontrivial manifolds. Topological data analysis methods can reveal the shared topological structure between neural manifolds and the task variables they represent, but may not to capture much of the geometric information including distance, angles, and curvature. 

![Overview of method to extract geometric features from neural activation manifolds. ](/method_overview.png)

We introduce a novel approach (see figure above) for studying the geometry of neural manifolds. This approach:
- computes an explicit parameterization of the manifolds, and
- estimates their local extrinsic curvature.  

We hope to open new avenues of inquiry exploring geometric neural correlates of perception and behavior, and provide a new means to compare representations in biological and artificial neural systems.



## 🏡 Installation ##

We recommend using Anaconda for easy installation and use of the method. To create the necessary conda environment, run:

```
conda env create -f environment.yml
conda activate neuralgeom
```
