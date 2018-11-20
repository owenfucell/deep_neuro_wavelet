# deep_neuro_wavelet
Continuation of the work of 
[Jannes Schafer](https://github.com/schanso/deep_neuro). 

**deep_neuro_wavelet**  explores the Gray dataset using a
[convolutional neural network](http://yann.lecun.com/exdb/publis/pdf/lecun-99.pdf) 
trying to predict either stimulus class or response type from different brain 
regions at different points in time.
It also transforms the raw data from MATLAB to NumPy-ready and applies the wavelet decomposition as well as dealing with the analysis and visualization of the results 


### Installation
To get started, create a project directory using `mkdir my_project/` and change
into it (`cd my_project/`). Then clone this repository using 
`git clone https://github.com/rnoyelle/deep_neuro_wavelet.git`. Once cloned, change into 
the directory (`cd deep_neuro/`) and source the environment generator file:

`. generate_environment.sh`

This will set up your folder structure (see below) and install all required packages. 
Move raw data into `data/raw/` and you should be ready to go.

```
my_project
|___data
|   |___raw
|
|___scripts
|   |____params
|   |___deep_neuro (git repo)
|       |___lib
|           ...
|
|___results
    |___training
    |___pvals
    |___summary
    |___plots
```


### Train classifier
To train a classifier, `cd` into your `scripts/deep_neuro/` directory. Set the processing parameters in `param_gen.py` and source 
the submit file using 

`. submit_training.sh` 

The job will be submitted to the 
cluster and processed once the resources are available.

How pre-processing using the matnpy module works is that :
it cuts every trial of a given session into five intervals:
* pre-sample (500 ms before stimulus onset)
* sample (500 ms after stimulus onset)
* delay (500 ms after stimulus offset)
* pre-match (500 ms before match onset)
* match (500 ms after match onset)

It selects channels into 6 groups :
* Visual cortex
* Motor cortex 
* Prefrontal cortex
* Parietal cortex
* Somatosensory cortex
* Visual + Motor + Somatosensory + Prefontal + Parietal cortex


### Get results
To get results, just `cd` into the `scripts/deep_neuro/` directory and source 
the submit file using :

`. get_results.sh` 

This will generate a summary file in 
`results/summary/` and a file of pvalues in 
`results/pvals/`. 

### Visualize results
To visualize results, run the jupyter notebook `graph_from_pval.ipynb` 

