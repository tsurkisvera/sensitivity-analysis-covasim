# Sensitivity Analysis of the Covasim Agent-Based Model

This repository contains scripts for the research paper  
*"Sensitivity Analysis of the Covasim Agent-Based Model: The Role of Synthetic Population Parameters"* (in preparation; DOI will be added later).

---

## Pipeline

To sample parameters for a synthetic population, use:  
- `sample_parameters_autoencoders.py` 

or 
- `sample_parameters_multiplicators.py`  

Then use `run_simulations.py` to obtain epidemiological results and `compute_indices.py` to conduct the sensitivity analysis.  

**Note:**

 All scripts should be run as:
```bash
python script folder_name additional_parameter
```
where `folder_name` is the name you chose for the folder containing your sampled parameters.

For **sampling scripts**, the additional parameter is the number of points in the parametric space to be involved in the analysis (not equal to the total number of evaluations; also, the balance properties of Sobol' points require the number of points to be a power of 2).  

See [Saltelli et al., 2010](https://www.andreasaltelli.eu/file/repository/PUBLISHED_PAPER.pdf) for details.

#### Example:
```bash
python sample_parameters_multiplicators.py test_folder 1024
```

For **running simulations**, the additional parameters are the folder name and the number of parameters involved in the analysis.  

#### Example:
```bash
python run_simulations.py test_folder 1024 8
```

For **computing indices**, no additional parameters are needed, but the problem dictionary should match the one defined in the sampling script.

#### Example:
```bash
python compute_indices.py test_folder
```
---

## Parameter Ranges

The ranges from which to sample are defined manually in the dictionary `'problem'` (sampling scripts). They were obtained by processing real-world data and utilizing autoencoders to sample from their latent spaces. This procedure can be found in the `data_processing` folder.

---

## Working with Results  

An example of dealing with indices once they are obtained can be found in the `plotting.ipynb` notebook.

---

## Important  

Please, use the **SALib version provided here**, as some minor changes were implemented to handle missing data (the case when not all synthetic populations are constructed successfully).


The `covasim` folder is the Covasim version 3.1.6 cloned from github at https://github.com/InstituteforDiseaseModeling/covasim and slightly changed to easily take a synthetic population from synthpops module as an input parameter.


The `synthpops` folder is the Synthpops version 1.10.5 cloned from github at https://github.com/InstituteforDiseaseModeling/synthpops and slightly changed to avoid infinite loops while constructing households. 

---

**For any questions, feel free to reach out!** 

 veratsurkis21 at gmail.com