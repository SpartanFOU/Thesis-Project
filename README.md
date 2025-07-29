# Bachelor's Project

This repository contains the code and experiments for bachelor's thesis project.

# Contact information
Mykyta Zaizzhai
zaizzmyk@cvut.cz


## Installation and Setup

### 1. Create Virtual Environment

First, create a Python virtual environment to isolate project dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

Install all required packages from the requirements file:

```bash
pip install -r requirements.txt
```

### 3. Export Dependencies (for developers)

If you add new packages during development, update the requirements file:

```bash
pip freeze > requirements.txt
```

## Usage

### Main Program Execution

The main program should be executed through **`main.ipynb`**. This Jupyter notebook contains all the primary code and analysis for the project.

**Important:** All code cells in `main.ipynb` should be executed in order to reproduce the complete analysis and results.

### Experiments and Models

- **Experiments**: To view all models that were experimented with during development, check the notebook located at:

  ```
  src/autoregression/experiments.ipynb
  ```

- **Model Building Scripts**: All model building and implementation scripts are stored in:
  ```
  src/autoregression/models/
  ```

## Troubleshooting

### Import Issues

If you encounter problems importing custom scripts, try adjusting the import paths:

**If imports are not working, try removing `src` from the import path:**

```python
# Instead of:
from src.autoregression import module_name

# Try:
from autoregression import module_name
```

**If that doesn't work, try adding `src` to the import path:**

```python
# Instead of:
from autoregression import module_name

# Try:
from src.autoregression import module_name
```

### Alternative Solution for Import Issues

You can also add the project root to your Python path:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))
```

## Project Structure

```
├── main.ipynb                          # Main program execution
├── requirements.txt                    # Project dependencies
├── src/
│   └── autoregression/
│       ├── experiments.ipynb          # Model experiments
│       └── models/                    # Model building scripts
└── README.md                          # This file
```

## Getting Started

1. Clone this repository
2. Follow the installation steps above
3. Open `main.ipynb` in Jupyter Notebook or JupyterLab
4. Execute all cells in order
5. Explore `src/autoregression/experiments.ipynb` for detailed model comparisons

## Requirements

- Python 3.11
- Jupyter Notebook/JupyterLab
- All dependencies listed in `requirements.txt`
