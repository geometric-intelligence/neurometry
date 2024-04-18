[![Test](https://github.com/geometric-intelligence/neurometry/actions/workflows/test.yml/badge.svg)](https://github.com/geometric-intelligence/neurometry/actions/workflows/test.yml)
[![Lint](https://github.com/geometric-intelligence/neurometry/actions/workflows/lint.yml/badge.svg)](https://github.com/geometric-intelligence/neurometry/actions/workflows/lint.yml)
[![Doc](https://img.shields.io/badge/docs-website-brightgreen?style=flat)](https://geometric-intelligence.github.io/?badge=latest)
[![Codecov](https://codecov.io/gh/geometric-intelligence/neurometry/branch/main/graph/badge.svg)](https://app.codecov.io/gh/geometric-intelligence/neurometry)
[![Python](https://img.shields.io/badge/python-3.11+-blue?logo=python)](https://www.python.org/)

<img width="1188" alt="Screen Shot 2024-04-05 at 8 50 36 PM" src="https://github.com/geometric-intelligence/neurometry/assets/8267869/f24ddbf2-78ce-4896-9417-ed966316af2e">

**Neurometry** is a computational framework to quantify geometric intelligence in natural and artificial brains. Neurometry provides functionalities to analyze the geometric structures underlying computation in neural systems - neural representations and neural manifolds.

This repository contains the official PyTorch implementation of the papers:
- **Quantifying Extrinsic Curvature in Neural Manifolds**. CVPR Workshop on Topology, Algebra and Geometry 2023.
[Francisco Acosta](https://web.physics.ucsb.edu/~facosta/), [Sophia Sanborn](https://www.sophiasanborn.com/), [Khanh Dao Duc](https://kdaoduc.com/), [Manu Mahdav](https://www.manusmad.com/) and [Nina Miolane](https://www.ninamiolane.com/).
- **Relating Representational Geometry to Cortical Geometry in the Visual Cortex**. NeurIPS Workshop on Unifying Representations in Neural Models 2023.
[Francisco Acosta](https://web.physics.ucsb.edu/~facosta/), [Colin Conwell](https://colinconwell.github.io/), [Sophia Sanborn](https://www.sophiasanborn.com/), [David Klindt](https://david-klindt.github.io/) and [Nina Miolane](https://www.ninamiolane.com/).


The neural manifold hypothesis postulates that the activity of a neural population forms a low-dimensional manifold within the larger neural state space, whose structure reflects the structure of the encoded task variables. Many dimensionality reduction techniques have been used to study the structure of neural manifolds, but these methods do not provide an explicit parameterization of the manifold, and may not capture the global structure of topologically nontrivial manifolds. Topological data analysis methods can reveal the shared topological structure between neural manifolds and the task variables they represent, but may not to capture much of the geometric information including distance, angles, and curvature.

![Overview of method to extract geometric features from neural activation manifolds. ](/method_overview.png)

We introduce a novel approach (see figure above) for studying the geometry of neural manifolds. This approach:
- computes an explicit parameterization of the manifolds, and
- estimates their local extrinsic curvature.

We hope to open new avenues of inquiry exploring geometric neural correlates of perception and behavior, and provide a new means to compare representations in biological and artificial neural systems.



## 🏡 Installation ##

We recommend using Anaconda for easy installation and use of the method. To create the necessary conda environment, run:

```
conda create -n neurometry python=3.11.3 cmake boost -c conda-forge -y
conda activate neurometry
pip install -e '.[all]'
```

If cuda is available, run instead:
```
conda create -n neurometry python=3.11.3 cmake boost -c conda-forge -y
conda activate neurometry
pip install -e '.[all,gpu]'
```

## 🌎 Bibtex ##

If this code is useful to your research, please cite:

```
@inproceedings{acostaQuantifyingExtrinsicCurvature2023,
  title = {Quantifying {{Extrinsic Curvature}} in {{Neural Manifolds}}},
  booktitle = {Proceedings of the {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}}},
  author = {Acosta, Francisco and Sanborn, Sophia and Duc, Khanh Dao and Madhav, Manu and Miolane, Nina},
  year = {2023},
  pages = {610--619},
  urldate = {2023-07-07},
  langid = {english},
  file = {/Users/facosta/Zotero/storage/BUNYT2IF/Acosta et al. - 2023 - Quantifying Extrinsic Curvature in Neural Manifold.pdf}
}
```

