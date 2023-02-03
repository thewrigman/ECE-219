# Readme

##Problems 1-7, 9
Note the code for problems 1-7 and 9 can be found in Problem1-7 and 9.ipynb. Much of the code itself relies on functions defined in utils.py. 
Running this code is fairly simple—it only involves running the cells and making sure the functions from utils.py are imported.

##Problem 8
The code for grid search and running the model for the top 5 can be found in gridsearch.py and top5.ipynb respectively. Much of the code itself relies on functions defined in utils.py.
In order to run this code, note that there is a section on line 124 as such:
            cachedir = mkdtemp(
                dir='/Users/ineshchakrabarti/ECE-219/')
On this line, change the path for dir to some path easily accessible to be used as the cache for faster completion of the grid search. Note that unless this is changed, the code will not run. 
top5.ipynb simply runs the top 5 models found in gridsearch.py

## Word Embeddings
For word embeddings we have the functions to train the model and optimize the hyperparameters in Problem11Pipline.py. Our implementation for the stochastic consensus based optimization method described by "A consensus-based global optimization method for high dimensional machine learning problems" is located in stochastic_optimizer.py along with a simple non-convex function that we used to test the optimizer and understand how its parameters affected its optimization.

The code we used for Problem 11 and the various methods for the optimizing the hyperparameters we discussed is located in Problem11.ipynb. The only execption is the code for gradient descent and grid search, those were early expermimentation code that we unfortunately inadverdently deleted.

Likewise the code for Problem 12 and 13 are located in Problem12.ipynb and Problem13.ipynb respectively.
