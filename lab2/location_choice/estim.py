# Simplified version of logit estimation of location choice model in recycling project

# Run one of the following to install optimagic if not already installed
# !pip install optimagic
# !conda install -c conda-forge optimagic

#%% LOAD MODULES
import numpy as np
import pandas as pd
import estimagic as em
from load_data import *

pd.options.display.max_rows = 500

vars_loc = ['grdpe', 'popdensity']
vars_indiv = ['Eduy', 'Age', 'Employee']

# full dataset
data = load_data(vars_loc=vars_loc, vars_indiv=vars_indiv, verbosity=1)
# reduced dataset
# data = load_data(vars_loc=vars_loc, vars_indiv=vars_indiv, verbosity=1, nloc=5, nindiv=1000)

data_indiv = data.data_indiv
data_loc = data.data_loc
choice_var = data.var_ind_choice

# sort locations by locid
data_loc = data_loc.sort_values(by='locid')
#%%
# Conditional logit model with all

# form parameters structure
# variables: all data_loc vars and all data_indiv*data_loc cross effects
coef_labels = vars_loc + [a + '_' + b for a in vars_loc for b in vars_indiv]
param = pd.DataFrame(
    data=[[0.001]] * len(coef_labels),
    columns=["value"],
    index=coef_labels,
)


def u(param, data_indiv, data_loc):
  # utilities at each location for each individual
  # all data_loc variables and all data_indiv*data_loc cross effects
  # in a linear combination with parameters
  # (dimension 0=indid, 1=locid)
  # convert to numpy arrays
  ind = np.array(data_indiv[vars_indiv])
  loc = np.array(data_loc[vars_loc])
  # number of location attributes
  m = len(vars_loc)
  # convert param value to numpy arrays
  p = np.array(param['value'])
  # combine indiv and loc attributes in 4-dim tensor
  # values in ind and loc are multiplied
  ind_loc_attr = np.einsum('ij,kh->ikhj', ind, loc)
  # select and shape cross-effect coefs
  p1 = p[m:].reshape((m, -1))
  # weight sum of attributes for each indiv-loc combination
  ind_loc = np.einsum('ijkh,kh->ij', ind_loc_attr, p1)
  # add loc characteristics
  p1 = p[:m]
  loc_const = np.einsum('ij,j->i', loc, p1)
  return ind_loc + loc_const[np.newaxis, :]


def chpr(param, data_indiv, data_loc):
  # choice prbabilities for each location for each individual
  # (dimension 0=indid, 1=locid)
  gamma = 1.
  X = u(param, data_indiv, data_loc)        # dims: indid,locid
  mm = np.amax(X, axis=1, keepdims=True)    # maximum along axis
  xx = (X - mm) / gamma                     # demax
  r1 = np.exp(xx)
  r2 = r1.sum(axis=1, keepdims=True)        # sum of exp along given axis
  return r1 / r2


def loglike(param, data_indiv, data_loc):
  # log likelihood of the sample
  # assuming that locid is from 0 to nloc-1
  #              (data_loc is sorted by locid)
  pr = chpr(param, data_indiv, data_loc)
  # index with the choice, compute indiv loglike
  iloglike = np.log(pr[np.arange(pr.shape[0]), data_indiv[choice_var]])
  # return average log-likelihood and contributions
  return {'contributions': iloglike, 'value': np.sum(iloglike) / len(data_indiv)}

l = loglike(param, data_indiv, data_loc)

#%%
# Rely on estimagic to estimate the model
res = em.estimate_ml(
    loglike=loglike,
    params=param,
    optimize_options={
        "algorithm": "scipy_lbfgsb",
        "algo_options": {
            "convergence.relative_criterion_tolerance": 1e-14,
            "stopping.max_iterations": 5_000,
        },
    },
    loglike_kwargs={
        "data_indiv": data_indiv,
        "data_loc": data_loc
    },
)
# display results
print(res.summary().round(4))

# %%
