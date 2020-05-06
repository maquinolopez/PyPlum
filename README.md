# Python implementation of Plum.
This is the python implementation of Plum. It creates age-depth models using 210Pb data, and the autoregressive gamma process presented in Blaauw and Christen (2011). PyPlum is the full implementation of Aquino-López et al. (2018). This implementation allows users to incorporate 210Pb to longer age-depth models (adding radiocarbon and calendar dates).

**cite: https://doi.org/10.1007/s13253-018-0328-7**

```diff
- this is the Python version of Plum, for the official R version see rplum.
```

## Requirements:
- Python3
- Numpy
- Scipy
- matplotlib
- pyTwalk (see: visit: https://www.cimat.mx/~jac/twalk/ for package and installing instructions)
- sklearn (this is use to estimate the number of samples to be used as supported 210Pb no needed if using 226Ra or provide the core file with this value)

## Retting folders
This code can be used as a Python module by putting the file in the core directory. I recommend to create a PyPlum folder in your ~/Documents folder (this is the default folder).

File PyPlum.py should be place in the folder where Cores and Calibration curves are located and python working directory has to be this same folder.

## File structure
Core data should be in a **csv** file with the following format:

|  ID  | Depth |   Density   | Total 210Pb | SD Total 210Pb | Thickness | Total 226Ra |  SD Total 226Ra | Info |
| :--- | :---- | :---------- | :---------- | :------------- | :-------- | :---------- | :-------------- | :--- |


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

This file (_Core\_name.csv_) should be inside a folder named with the same Core name. This Core folder should be in the folder where _PyPlum.py_ and _Calibration Curves_ folder are.

If radiocarbon or calendar years are available, add a file with the same name as the Core file but adding _\-C_ such that you have a file named _Core\_name\-C.csv_. This file should have the following format:

|  ID  | BP Age | Age error | Depth |  cc  |
| :--- | :----- | :-------- | :---- | :--- |

- **ID:**         
	Sample's ID
- **BP Age:**			
	Sample's age on **Before Present** format
- **Age error:**
	Standar deviation of sample's age
- **Depth:**
	Sample's depth
- **cc:** Calibration curve: 0. Calendar dates, 1. IntCal13.14C, 2. Marine13.14C, 3. SHCal13.14C

Plum will automatically identify the presence of the _Core\_name\-C.csv_ and run the appropriate model.

## Running instructions:

To load the package use:

		import PyPlum

Loading data and preparing for analysis:

		Core_name = PyPlum.PyPlum('Core_name')

This will load the data and set everything for plum to run. It will display the data that _PyPlum_ will use and also

To run the analysis:

	Core_name.runPlum()

This will run the _PyPlum_ using default settings.

## Settings

- **Core:**  Core name
- **dirt:**  Core folder location
- **Dircc:**  Calibration curve folder
- **thick:**  length of the autoregressive gamma process sections
- **n_supp:**  Number samples to be used exclusively for inferring  supported 210Pb
- **mean_m:**  Prior mean of memory parameter (used by the autoregressive gamma process)
- **shape_m:**  Prior shape of memory parameter (used by the autoregressive gamma process)
- **mean_acc:** Prior mean of alpha parameters (used by the autoregressive gamma process)
- **shape_acc:** Prior shape of alpha parameters (used by the autoregressive gamma process)
- **fi_mean:**  Prior mean of prior 210Pb influx
- **fi_shape:**  Prior shape of 210Pb influx
- **s_mean:**		Prior mean of supported 210Pb
- **s_shape:**	Prior shape of supported 210Pb
- **intv:**			Length of credible interval
- **Ts_mod:**		If _True_ radiocarbon likelihood will be constructed using the T model, if _False_ the normal model will be used
- **iterations:** Bumber of final MCMC iteration
- **burnin:**		Size of burn-in  
- **thi:**			Thinning of the MCMC
- **cc:**				Calibration Curve (name of file containing the calibration curve)
- **ccpb:**			Postbomb Calibration curve
- **showchrono:** If _True_ the chronology will be display after the running the model, if _False_ chronology will not be display. In both cases chronology will be save as a _pdf_ file
- **g_thi:**  Size of radiocarbon globes in plot
- **Al:**			Detection limit, this variable limits the chronology (_see: Aquino-López et al. 2018 for details)_
- **seed:**  Seed for the random variables (used to replicate results). If not changed random seed will be used.
- **d_by:**  Length of sections used to create age file.



## rplum, R's official version

rplum has being accepted into CRAN (the R reposatories). For downloading this version use

`install.packages('rplum')`

`library(rplum)`

`Plum()`
