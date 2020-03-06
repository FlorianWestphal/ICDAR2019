## Raw Data

This folder contains the raw data collected for this study. Each
of the CSV files contains the recognition accuracy results for the
evaluated recognition strategies and all the ten randomly sampled, 
balanced training sets of size 10, 30, 50, 70, 100, 250, 500, and 1000.
The results for the MNIST datasets can be found in the CSV files 
`final_mnist_1.csv` and `final_mnist_2.csv`, while the results for the
KMNIST dataset can be found in the CSV files `final_kmnist_1.csv` and
`final_kmnist_2.csv`. 

The results for the different strategies evaluated on the test set containing
only 1500 images can be found in the files `final_mnist_1.csv` and 
`final_kmnist_1.csv`, while the results evaluated on the complete test sets can
be found in the files `final_mnist_2.csv` and `final_kmnist_2.csv`.
The column names of the CSV files encode the recognition strategy, as well as 
the number of training samples used as follows: `<strategy>_<samples>`. The
strategy names in the CSV files map to the strategy names in the paper as 
follows:

| Name in Paper | Name in CSV |
| ---- | ---- |
| $HED_{kNN}$ | `ged_knn` |
| $\widehat{HED}_{kNN}$ | `sged_knn` |
| $\widehat{HED}_{NN}$ | `sged_nn` |
| $\widehat{HED}_{NN+}$ | `sged_nn_train` |
| $S$ | `normal` |
