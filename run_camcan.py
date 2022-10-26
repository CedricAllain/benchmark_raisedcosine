# %%
from alphacsc import GreedyCDL
from alphacsc.datasets.camcan import load_data

subject_id = 'sub-CC620264'
sfreq = 150.
load_params = dict(sfreq=sfreq, n_jobs=5, n_splits=10)

cdl_params = {
    # Shape of the dictionary
    'n_atoms': 20,
    'n_times_atom': int(round(sfreq * 1.0)),
    # Request a rank1 dictionary with unit norm temporal and spatial maps
    'rank1': True,
    'uv_constraint': 'separate',
    # apply a temporal window reparametrization
    'window': True,
    # at the end, refit the activations with fixed support
    # and no reg to unbias
    'unbiased_z_hat': True,
    # Initialize the dictionary with random chunk from the data
    'D_init': 'chunk',
    # rescale the regularization parameter to be a percentage of lambda_max
    'lmbd_max': "scaled",  # original value: "scaled"
    'reg': 0.2,
    # Number of iteration for the alternate minimization and cvg threshold
    'n_iter': 100,  # original value: 100
    'eps': 1e-5,  # original value: 1e-4
    # solver for the z-step
    'solver_z': "lgcd",
    'solver_z_kwargs': {'tol': 1e-3,  # stopping criteria
                        'max_iter': 100000},
    # solver for the d-step
    'solver_d': 'alternate_adaptive',
    'solver_d_kwargs': {'max_iter': 300},  # original value: 300
    # sort atoms by explained variances
    'sort_atoms': True,
    # Technical parameters
    'verbose': 1,
    'random_state': 0,
    'n_jobs': 5
}


def run_csc(subject_id, load_params, cdl_params):
    X_split, info = load_data(subject_id=subject_id, **load_params)
    # cdl = GreedyCDL(**cdl_params)
    # cdl.fit(X_split)
    n_splits, n_channels, n_times = X_split.shape
    X = X_split.swapaxes(0, 1).reshape(n_channels, n_times * n_splits)
    z_hat = cdl.transform(X[None, :])
    
    return X_split, info
