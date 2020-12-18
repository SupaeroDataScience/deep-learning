# Dimensionality Reduction

* [Home](https://supaerodatascience.github.io/deep-learning/)
* [Github repository](https://github.com/SupaeroDataScience/deep-learning/)
* [Initial Github repository](https://github.com/jfabrice/ml-class-dimensionality-reduction)


## Algorithms in Machine Learning, ISAE-SUPAERO

This repository contains course materials for my Machine Learning class on the topic of Dimensionality Reduction:
- <em>dimensionality_reduction_class.pdf</em> : course presentation with teaching content
- <em>dimensionality_reduction_class.ipynb</em> : Python notebook illustrating the course notions
- <em>dimensionality_reduction_class-empty.ipynb</em> : same Python notebook to follow with students during class


### Option 1: Working with Google Colab

To follow the notebooks with Google Colab, simply go to https://colab.research.google.com/. Import a new notebook from GitHub, search for "jfabrice" and open one of the notebooks of this class (ml-class-dimensionality-reduction), for example dimensionality_reduction_class-empty.ipynb. Then click on "Copy to Drive" to be able to execute it. The first section of the notebook is there to initialize the environment from Google Colab.


### Option 2: Working locally - Setting up Anaconda environment

To setup the Anaconda environment with required dependencies, execute the following instructions in Anaconda prompt or Linux shell.

```shell
# Clone this github repository on your machine
git clone https://github.com/jfabrice/ml-class-dimensionality-reduction.git

# Change working directory inside the repo
cd ml-class-dimensionality-reduction

# Create a new virtual environment
conda create -n dimreductionenv python==3.6

# Activate the environment
## For Linux / MAC
source activate dimreductionenv
## For Windows
activate dimreductionenv

# Install the requirements
pip install -r requirements.txt
```
