```markdown
# DPMM Wrapper for space-ai

This module isolates anomaly detection logic using `torch-dpmm`, which requires specific versions of numpy, sklearn, etc.

## When to use this
Use this if you're running anomaly detection models based on DPMM in a separate environment due to dependency constraints.

---

## Required Environment
<<<<<<< HEAD

This module **requires Python 3.10.15**. We recommend creating a conda or virtual environment named `dpmm_env`.

### Install Python 3.10.15

**Using conda:**  
`conda install python=3.10.15`

**Using pyenv (Linux/Mac):**  
`pyenv install 3.10.15`  

Verify installation:  
`python --version`

### Create environment with conda (Linux/Mac/Windows):

`conda create -n dpmm_env python=3.10.15`  
`conda activate dpmm_env`  
`pip install -r requirements_dpmm.txt`

### Or create environment with pip and venv:

**On Linux/Mac:**  
`python3.10 -m venv dpmm_env`  
`source dpmm_env/bin/activate`  
`pip install -r requirements_dpmm.txt`

**On Windows:**  
`python3.10 -m venv dpmm_env`  
`.\dpmm_env\Scripts\activate`  
`pip install -r requirements_dpmm.txt`

### Installation on Mac (Apple Silicon - optional)

If you're using a Mac with Apple Silicon (M1/M2), ensure compatibility by installing dependencies from conda:

`conda create -n dpmm_env python=3.10.15`  
`conda activate dpmm_env`  
`conda install pytorch torchvision torchaudio -c pytorch`  
`pip install -r requirements_dpmm.txt`

---

## How to run manually (optional)

`python run_dpmm.py test.csv train.csv output.csv likelihood Full 100 50 0.8`

---

## Usage example in Python

When initializing the detector, you need to specify the path to your virtual environment's Python executable:

```python
from spaceai.models.anomaly.dpmm_detector import DPMMWrapperDetector

detector = DPMMWrapperDetector(
    mode="likelihood",       # or "new_cluster"
    model_type="Full",
    K=100,
    num_iterations=50,
    lr=0.8,
    python_executable="/path/to/your/env/dpmm_env/bin/python"  # Replace with your actual path
)
```
to see your path you can run the following command in your terminal when your dpmm_env is activated:
```bash
which python
```

=======
We recommend creating a conda environment named `dpmm_env`:

### Create it with conda:
```bash
conda create -n dpmm_env python=3.10
conda activate dpmm_env
pip install -r requirements_dpmm.txt
```

### Or with pip and venv:
```bash
python -m venv dpmm_env
source dpmm_env/bin/activate  # or .\dpmm_env\Scripts\activate on Windows
pip install -r requirements_dpmm.txt
```

---

##  How to run manually (optional)
```bash
python run_dpmm.py test.csv train.csv output.csv likelihood Full 100 50 0.8
```
>>>>>>> 57fe8dc (versione con dpmm)

## File Structure
```
external/
└── dpmm_wrapper/
    ├── run_dpmm.py
    ├── dpmm_core.py
    ├── benchmark_utils.py
    ├── requirements_dpmm.txt
    └── README.md
```

---
<<<<<<< HEAD
```
=======

>>>>>>> 57fe8dc (versione con dpmm)
## Used from space-ai
The module is launched from within `space-ai` using the `DPMMWrapperDetector`, which runs the wrapper in subprocess. See `examples/example_dpmm_wrapper.py`.
```
