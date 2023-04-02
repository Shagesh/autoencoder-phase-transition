# README

This code implements a dictionary learning method to reconstruct sparse signals. The main script, `Perm_inv_online_dictionary_learn.py`, implements the dictionary learning algorithm and tests it using a synthetic dataset. The synthetic dataset is generated using the `make_sparse_coded_signal` function from `sklearn.datasets`.

The `Perm_inv_online_dictionary_learn.py` script includes several functions, including:

- `perm_free_D(D)`: a function that performs permutation-free dictionary normalization.
- `perm_free_DS(D, X)`: a function that performs permutation-free dictionary and signal normalization.
- `make_sparse_coded_signal1(n_samples, n_components, n_features, n_nonzero_coefs, random_state=None, data_transposed="warn")`: a modified version of `sklearn.datasets.make_sparse_coded_signal` function that returns the dictionary, code and signal with the correct shapes.
- `dict_reconstruction(Y, n_comp)`: a function that performs the dictionary reconstruction.
- `dict_reconstruction_online(Y, n_comp, n_nonzero_coefs, n_feat, n_samp)`: a function that performs the dictionary reconstruction in an online manner.

For an example on how to use the code, you can refer to the `Perm_inv_online_dictionary_learn.ipynb` jupyter notebook. You can adjust the hyperparameters of the method by changing the values of the variables at the beginning of the notebook.

The code generates a phase transition plot that shows the relationship between the sparsity of the signal and the number of measurements required to reconstruct it.

This code requires `sklearn`, `numpy`, `scipy`, `matplotlib`, `torch`, `torchvision`, `importlib` and `tqdm`.

The next step in this project is to analyze the stability of the planted dictionary learning solution in an autoencoder setting.
