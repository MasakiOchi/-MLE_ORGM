# MLE_ORGM
Codes for the maximum-likelihood estimate of the vertex sequence using ORGM

Sourse code to estimate the optimal vertex sequence of a network using the maximum-likelihood estimate (MLE) based on ordered random graph model (ORGM) [1].

## MLE.ipynb

Sample code when we apply this algorithm to a network.
Input network data should be implimented in graph-tool [2] .
The optimal vertex sequence and model parameters of the ORGM is obtained by "MLE_ORGM" in "MLE_algorithm.py".
The heatmap of the adjacency matrix based on the inferred vertex sequence is obtained by "plot_heatmap" in "MLE_tools.py".


## MLE_algorithm.py

Code of the MLE.

- Parameters
  - g : network to infer the vertex sequence implimented by graph-tool
  - K : Number of sine waves to consider in the envelope function.
  - beta, eta, sample_prob : Hyperparameters
  - n_sm : Number of initial states for {a_k} to consider in order to avoid being trapped into the local minima. The state with the highest likelihood is employed.
- Return
  - vertex_sequence_max : Vertex sequence of the network
  - a_max : Model parameter {a_k}
  - pin_max, pout_max : Model parameters p_in and p_out
  - l_max : Achieved value of log-likehood function

## MLE_tools.py

Tools to impliment "MLE_algorithm.py" and draw heatmaps.
