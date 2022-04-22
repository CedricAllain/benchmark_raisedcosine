# first line: 161
@memory.cache(ignore=['n_jobs'])
def run_multiple_em_on_synthetic(simu_params, simu_params_to_vary,
                                 em_params, em_params_to_vary,
                                 sfreq=150., n_drivers=1, n_jobs=6,
                                 save_results=False):
    """Run several EM in parallel for multiple simulation parameters
    combinations and for multiple EM parameters combinations

    Parameters
    ----------
    simu_params : dict
        dictionary of all simulation parameters and their values

    simu_params_to_vary : dict
        dictionary of all simulation parameters we want to vary (keys) and
        their corresponding set of values

    em_params : dict
        dictionary of all EM parameters and their values

    em_params_to_vary : dict
        dictionary of all EM parameters we want to vary (keys) and
        their corresponding set of values

    sfreq : int | None
        sampling frequency used to create a grid between kernel's lower and
        upper to pre-compute kernel's values
        if None, the kernel will be exactly evaluate at each call.
        Warning: setting sfreq to None may considerably increase computational
        time.
        default is 150.

    n_jobs : int
        The maximum number of concurrently running jobs
        default is 6

    Returns
    -------
    pandas.DataFrame

    """
    # get all parameters combination for data simulation
    combs_simu = list(itertools.product(*list(simu_params_to_vary.values())))
    # get all parameters combination for EM algorithm
    combs_em = list(itertools.product(*list(em_params_to_vary.values())))
    # get maximum duration, to simulate with
    T_max = em_params_to_vary['T'].max()

    # define dataframe to stock results
    df_res = pd.DataFrame()

    # run in parallel
    if n_jobs > 1:
        new_rows = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(procedure)(this_comb_simu, combs_em, T_max,
                               simu_params, simu_params_to_vary,
                               em_params, em_params_to_vary, sfreq, n_drivers)
            for this_comb_simu in combs_simu)
    # run linearly
    else:
        new_rows = []
        for this_comb_simu in tqdm(combs_simu):
            new_row = procedure(this_comb_simu, combs_em, T_max,
                                simu_params, simu_params_to_vary,
                                em_params, em_params_to_vary, sfreq, n_drivers)
            new_rows.append(new_row)

    for this_new_row in new_rows:
        df_res = df_res.append(this_new_row, ignore_index=True)

    if save_results:
        # save dataframe as csv
        path_df_res = SAVE_RESULTS_PATH
        if not path_df_res.exists():
            path_df_res.mkdir(parents=True)

        df_res.to_csv(path_df_res / 'results_em_synthetic.csv')

    return df_res
