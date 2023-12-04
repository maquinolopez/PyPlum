# Python Implementation of Plum (PyPlum)

PyPlum is a comprehensive Python implementation of the Plum model, which generates age-depth models using 210Pb data through the autoregressive gamma process, as detailed in Blaauw and Christen (2011). This implementation, based on Aquino-López et al. (2018), extends 210Pb models by integrating radiocarbon and calendar dates.

**Citation**: For detailed methodology, refer to [Aquino-López et al., 2018](https://doi.org/10.1007/s13253-018-0328-7).

**Note**: PyPlum is the Python variant of Plum. For the official R version, see `rplum`.

## Prerequisites:

To use PyPlum, ensure you have the following installed:
- Python 3
- Numpy
- Scipy
- Matplotlib
- PyTwalk (Installation guide available at [PyTwalk](https://www.cimat.mx/~jac/twalk/))
- Sklearn (Required for estimating 210Pb samples; not needed for 226Ra or pre-provided core file values)

## Setting up the Environment

1. **Module Setup**: PyPlum can be used as a Python module. Place the `PyPlum.py` file in your project's core directory. We recommend creating a `PyPlum` folder in `~/Documents` (default location).
2. **File Location**: Ensure `PyPlum.py` is located in the directory containing Cores and Calibration curves. Set this directory as your working directory in Python.

## File Structure

Core data should be stored in a **csv** format as follows:

|  ID  | Depth |   Density   | Total 210Pb | SD Total 210Pb | Thickness | Total 226Ra |  SD Total 226Ra | Info |

- **ID:**             
	Sample's ID
- **Depth:**        
	Lower depth of sample
- **Density:**      
	Density of sample (g/cm^3)
- **Total 210Pb:**   
	Total 210Pb measurements (Bq/kg)
- **SD Total 210Pb:**
	Standard deviation of Total 210Pb measurements
- **Thickness:**
	Samples thickness (upper - lower depth)
- **Total 226Ra:**
	Total 226Ra measurements (Bq/kg)
- **SD Total 226Ra:**
	Standard deviation of Total 226Ra measurements
- **Info:**		  
	Core's information: First cell should contain sampling date and second cell can contain the number of samples to be used for only supported activity.  

Ensure the core data file (`Core_name.csv`) is placed in a dedicated Core folder, which is in the same directory as `PyPlum.py` and `Calibration Curves`.

For radiocarbon or calendar data, add a `Core_name-C.csv` file with the format:

|  ID  | BP Age | Age error | Depth |  cc  |

- **ID:**         
	Sample's ID
- **BP Age:**			
	Sample's age on **Before Present** format
- **Age error:**
	Standar deviation of sample's age
- **Depth:**
	Sample's depth
- **cc:** Calibration curve: 0. Calendar dates, 1. IntCal13.14C, 2. Marine13.14C, 3. SHCal13.14C

PyPlum will automatically detect the presence of `Core_name-C.csv` and adjust the model accordingly.

## Running Instructions

1. **Loading the Package**: 

   ```{python}
   import PyPlum
   ```
2. **Preparing for Analysis**:

   ```{python}
   Core_name = PyPlum.Plum('Core_name')
   ```
   This loads the data and prepares PyPlum for execution.
3. **Running the Analysis**:

   ```{python}
   Core_name.runPlum()
   ```
   This executes PyPlum with default settings.

## Settings

The following settings can be adjusted in PyPlum:

- **Core:** Core name
- **dirt:** Core folder location
- ...

For a detailed explanation of each setting, refer to the documentation.

## rplum, R's Official Version

For the R version of Plum, install `rplum` from CRAN:

```R
install.packages('rplum')
library(rplum)
Plum()
```

---

This revised README provides a clear, organized, and user-friendly guide for PyPlum users.