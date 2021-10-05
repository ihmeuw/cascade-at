def asymptotic_statistics(DB, n_samples = 500, table_name = '_doc_var_statistics'):
    """
    Use the dismod 'sample asymptotic' command to generate random samples of the model input variables, then compute the statistics for those samples.
    """
    with sqlite3.connect(DB.filename) as con:

        def types(covariate):
            if covariate.startswith('x_c_'): return 'Country-level'
            if covariate.startswith('x_'): return 'Study-level'    

        def mulcov_table_stats(stats, table_name = '_doc_mulcov_statistics'):
            mstats = stats.merge(DB.var, how='left', left_on='fit_var_id', right_on='var_id').merge(DB.covariate).merge(DB.rate, how='left').merge(DB.integrand, how='left')
            mstats['measure'] = [r if r else c for r,c in mstats.loc[:, ['rate_name', 'integrand_name']].values]
            mstats['type'] = [types(c) for c in mstats.covariate_name]
            mstats.rename(columns={'fit_var_id':'var_id', 'fit_value':'value', 'sam_std':'std'}, inplace=True)
            cols = ('var_id', 'measure', 'type', 'covariate', 'value', 'std')
            mstats = mstats.loc[:, cols]
            print (mstats)
            return mstats

        # Generate samples
        run_AT_commands(DB.filename, ['sample asymptotic 100'])
        # Compute input variable statistics
        sample = DB.sample
        samples = sample.groupby('var_id')
        mean = samples.var_value.mean()
        std = samples.var_value.std()
        fit = DB.fit_var.fit_var_value
        bias = mean - fit
        stats = pd.DataFrame(dict(fit_var_id = samples.nth(0).sample_id, sam_mean = mean, sam_std = std, sam_bias = bias))
        stats.to_sql(table_name, con, index=False, if_exists='replace', dtype={"fit_var_id":"integer primary key"})
        if 0:
            mstats = mulcov_table_stats(stats)
            mstats.to_sql(table_name+'_xxx', con, index=False, if_exists='replace', dtype={"var_id":"integer primary key"})
        return stats
