# Data and custom codes for Han et al. (2025)
---
This repository contains all custom code and data used to estimate influenza oral antiviral (oseltamivir and baloxavir marboxil) demand and impact across 186 countries worldwide. The original manuscript describing this work has been published as a preprint:

> Alvin X Han, Katina D. Hulme, Colin A. Russell. (2025). The global demand and potential public health impact of oral antiviral treatment stockpile for influenza pandemics.

---
## System Requirements & Installation
All codes were tested/ran on macOS Sonoma 14.1 and Red Hat Enterprise Linux 8.10 Ootpa. All Python codes run on Python (v3.12.2) and depend on the Python scientific stack:  

```
numpy (v1.26.4)
scipy (v1.12.0)
pandas (v2.21)
sciris (v3.1.3)
sklearn (v1.4.1.post1)
numba (v0.59.1)
matplotlib (v3.10.0)
seaborn (v0.13.2)
```

Install Python and dependencies using `miniconda`. See https://www.anaconda.com/docs/getting-started/miniconda/main for detailed instructions.  

All R codes run on R (v4.4.0) and require the following packages which can be installed by:
`install.packages(c('tidyverse', 'tidybayes', 'brms', 'broom', 'broom.mixed'))`

You can clone the GitHub repository by:
```
git clone https://github.com/AMC-LAEB/flu-antiviral-stockpile
```

---
## Demo
You can run a demo run of the transmission model that simulates the US 2017/2018 influenza season by:  
```
cd flu-antiviral-stockpile
python renewal.py simulate
```

Additional command flags can be found by `python renewal.py simulate --help`

---
## Instructions
1. All model simulations in the paper can be reran using the wrapper scripts ("run_*.py") in the scripts folder.  
2. All codes used to generate the figures in the manuscript can be found in the Jupyter Notebooks under the notebooks folder.  
3. The Bayesian hierarchical analyses can be reran by using mlm.R and mlm_av.R in the scripts folder.  

Please contact x.han@amsterdamumc.nl if you have any questions.
