from types import SimpleNamespace
import pandas as pd
import numpy as np

def load_data(vars_loc=[],
              vars_indiv=[],
              fname_mun='./data_mun.dta',
              fname_indiv='./data_indiv_sample.dta',
              verbosity=1,
              nloc=0,
              nindiv=0,
              var_indid='indid',
              var_ind_choice='locid_choice',
              var_ind_effort='recycling65',
              var_locid='locid',
              var_loc_rcost='rcost',
              var_loc_n='ppl',
              var_loc_effort='recycling'):
  ''' Loads the dataset prepared by State run_data4estimation.do
      Defaults are set to the names in our usual dataset in the usual files
  INPUT:  fname_mun - file name for location data
          fname_indiv - file name for individual data
          -- variable names --
          var_indid - individual id variable name
          var_ind_choice - location choice variable name (must correspond to values of var_locid)
          var_ind_effort - effort variable name
          vars_indiv - list of individual variables to be loaded
          var_locid - location id variable name
          var_loc_rcost - location recycling costs variable name
          var_loc_n - location population variable name
          var_loc_effort - location average effort variable name
          vars_loc - list of municipality variables to be loaded
          -- data thinning + verbosity --
          nloc - number of locations to be used when thinning the data (0 for all)
          nindiv - number of individuals to be used when thinning the data (0 for all)
          verbosity - print data info
  OUTPUT: namespace with the following entries:
          data_loc - panadas dataframe with location data
          data_indiv - pandas dataframe with individual data
          -- variable names --
          var_indid - individual id variable name
          var_ind_choice - location choice variable name (must correspond to values of var_locid)
          var_ind_effort - effort variable name
          vars_indiv - list of individual variables to be loaded
          var_locid - location id variable name
          var_loc_rcost - location recycling costs variable name
          var_loc_n - location population variable name
          var_loc_effort - location average effort variable name
          vars_loc - list of municipality variables to be loaded
          -- computed from indiv data --
          indiv_nu - mean of individual characteristics
          indiv_cov - covariance matrix of individual characteristics
          -- saved for reference --
          fname_mun - file name for location data
          fname_indiv - file name for individual data
  '''
  # Municipality level data
  if verbosity > 0:
    print(f'Loading municipality level dataset {fname_mun}')
  data = pd.read_stata(fname_mun)
  # required variables
  assert var_locid in data.columns, 'Location data does not have location ID variable'
  assert var_loc_rcost in data.columns, 'Location data does not have cost of recycling variable'
  assert var_loc_n in data.columns, 'Location data does not have population size variable'
  assert var_loc_effort in data.columns, 'Location data does not have recycling effort variable'
  # at least one municipality level variable is required, could be constant
  assert len(vars_loc) > 0, 'At least one amenity variable needs to be chosen, data load aborted'
  # select variables + sort
  data_loc = data[[var_locid, var_loc_rcost, var_loc_n, var_loc_effort] + vars_loc]
  data_loc = data_loc.sort_values(by=var_locid)
  data_loc = data_loc.drop_duplicates(subset=[var_locid], keep='first')

  # Indiv data
  if verbosity > 0:
    print(f'Loading individual level dataset {fname_indiv}')
  data = pd.read_stata(fname_indiv)
  assert var_indid in data.columns, 'Individual data does not have ID variable'
  assert var_ind_choice in data.columns, 'Individual data does not have location choice variable'
  assert var_ind_effort in data.columns, 'Individual data does not have recycling effort variable'
  # at least one individual level variable is required, could be constant
  assert len(vars_indiv) > 0, 'At least one individual characteristic variable needs to be chosen, data load aborted'
  data_indiv = data[[var_indid, var_ind_choice, var_ind_effort] + vars_indiv]
  data_indiv = data_indiv.sort_values(by=var_indid)
  data_indiv = data_indiv.drop_duplicates(subset=[var_indid], keep='first')

  # Recode location id and choice variable to start from 0
  assert data_indiv[var_ind_choice].isin(data_loc[var_locid]).all(), 'Some location choices are not among the location ids'
  locids = data_loc[var_locid].unique()
  index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(locids)}
  data_loc[var_locid] = data_loc[var_locid].map(index_mapping)
  data_indiv[var_ind_choice] = data_indiv[var_ind_choice].map(index_mapping)

  # Thinning the data
  # collapse number of locations to nloc if given
  if nloc > 0:
    # replace location codes with residuals
    data_loc[var_locid] = data_loc[var_locid] % nloc
    data_loc = data_loc.drop_duplicates(subset=[var_locid], keep='first')
    data_indiv[var_ind_choice] = data_indiv[var_ind_choice] % nloc
  # collapse number of individuals to nindiv if given
  if nindiv > 0:
    # just take the first nindiv records
    data_indiv = data_indiv.iloc[:nindiv]

  # Add index made from id variables
  data_loc['indx'] = data_loc[var_locid]
  data_loc.set_index('indx', inplace=True)
  data_indiv['indx'] = data_indiv[var_indid]
  data_indiv.set_index('indx', inplace=True)

  if verbosity > 0:
    print('\n####### Location data #######')
    print(data_loc.info())
    print(data_loc.describe())
    print('\n####### Individual sample data #######')
    print(data_indiv.info())
    print(data_indiv.describe())

  # fit mulnivariate normal on individual characteristics
  indiv_char = data_indiv[vars_indiv]
  indiv_nu = np.array(indiv_char.mean())
  indiv_cov = np.array(indiv_char.cov())

  # return namespace of datasets, statistics and variable names
  return SimpleNamespace(**{
      'data_loc': data_loc,
      'data_indiv': data_indiv,
      'sim_population': None,
      'indiv_nu': indiv_nu,
      'indiv_cov': indiv_cov,
      'fname_mun': fname_mun,
      'fname_indiv': fname_indiv,
      'var_indid': var_indid,
      'var_ind_choice': var_ind_choice,
      'var_ind_effort': var_ind_effort,
      'var_locid': var_locid,
      'var_loc_rcost': var_loc_rcost,
      'var_loc_n': var_loc_n,
      'var_loc_effort': var_loc_effort,
      'vars_indiv': vars_indiv,
      'vars_loc': vars_loc,
  })
