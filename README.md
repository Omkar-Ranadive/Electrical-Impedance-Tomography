# Electrical Impedance Tomography 
Collection of algorithms to optimize sensitivity volume for the problem of electrical impedance tomography by efficiently selecting a small subset of rows from a large matrix.

## Algorithms available 

### Genetic Algorithms
- Standard Genetic (algo_genetic.py)
- Ensemble Genetic (algo_genetic.py)
- Greedy Genetic (algo_genetic.py)

Genetic algorithms can be tested through run_ensembleGen.py and run_greedyGen.py files 

### Clustering Algorithms 
- Modified Clustering (algos.py)
- Custom KMeans (algo_custom_kmeans.py) 

Clustering algorithms can be tested through run_clustering_v{}.py files 

### Machine Learning Algorithms 
- Deep Q-Network (algo_dqn.py) 
- Attention-based Deep Network (algo_nn.py) 
- Transformer Model (algo_nn.py) 

ML algorithms can be trained through run_nn.py, run_transformer.py and then the predictions
can be obtained through run_nn_predict.py, run_transformer_predict.py 

### Other algorithms 
- Highest Magnitude Approach 
- Greedy Approach 
- Max Radial Coordinates Approach 
- Max radial coordinate approach with flipped signs
- Max radial coordinates with each axis as init centroid
- Random volume approach 
- Modified Cosine 
- Singular Value Approach 

These algorithms can be found inside algos.py and tested through run_multiple.py 

## Visualizations 
A variety of visualizations can be performed through vis_results.py and compare_sol.py 

## Data and Results 
The algorithms expect the data to be in form of numpy arrays, but the run functions load the 
data through .mat matrices. These matrices should be placed inside the data folder. All results 
are generated inside the exp folder. 

