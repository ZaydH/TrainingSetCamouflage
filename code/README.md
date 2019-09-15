## Prerequisites
Python 3.6.8 <br/>
Packages: numpy=1.16.4, scipy=1.1.0


## experiment_uniform_sampling_lr.py
script to run Uniform Sampling algorithm using logistic regression

`input_dir`: The directory containing the input data <br/>
`output`: Path prefix to the output file <br/>
`train_size`: Size of the training set to use <br/>
`thinking_budget`: Number of learners to train for search <br/>
`iterations`: Number of iterations to divide the budget <br/>
`loss`: The loss function <br/>
`seed`: The random seed <br/>
`homogeneous`: Flag for homogeneous logistic regression <br/>
`cpus`: The number of cpus to use <br/>
`print_vector`: Flag to print the final vector <br/>

python experiment_uniform_sampling_lr.py -i ../data/Toy_Dataset_1 -o ./uniform -n 100 -b 1000 --iterations 10


## experiment_beam_search_lr.py
script to run Beam Search algorithm using logistic regression

`input_dir`: The directory containing the input data <br/>
`output`: Path prefix to the output file <br/>
`train_size`: Size of the training set to use <br/>
`thinking_budget`: Number of learners to train for search <br/>
`beam_width`: Width of the beam <br/>
`children`: Number of children to consider for each sequence <br/>
`loss`: The loss function <br/>
`seed`: The random seed <br/>
`homogeneous`: Flag for homogeneous logistic regression <br/>
`cpus`: The number of cpus to use <br/>
`print_vector`: Flag to print the final vector <br/>

python experiment_beam_search_lr.py -i ../data/Toy_Dataset_1 -o ./beam -n 100 -b 1000 -w 10 --children 2

## load_learners.py
Contains methods to load a logistic regression learner.

## fit_learners.py
Contains methods to train a logistic regression learner.


## utility.py
Contains utility methods.