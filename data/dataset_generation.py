import pandas as pd 
import numpy as np
import random

### generate a dataset of n samples and standardize the variables x1,x2,x3
def generate_dataset(n_data=3000, n_variables=10, tau=5, n_xx=10, n_xy=3, n_yy=2, n_yx=5, seed=0):

    rng = np.random.default_rng(seed=seed)
    random.seed(seed)
    np.random.seed(seed)

    x = np.zeros((n_data,n_variables))
    y = np.zeros(n_data)

    # causal coefficients to x
    x_start = list(np.random.randint(low=0, high=n_variables, size=n_xx))
    x_end = list(np.random.randint(low=0, high=n_variables, size=n_xx))
    x_lag = list(np.random.randint(low=1, high=tau, size=n_xx))
    x_coeff = list(rng.uniform(-1,1,size=n_xx))

    for i in range(n_yx):
        x_start.append(n_variables)
        x_end.append(np.random.randint(low=0, high=n_variables))
        x_lag.append(np.random.randint(low=1, high=tau))
        x_coeff.append(rng.uniform(-1,1))

    #x_start = [3, 5]
    #x_end = [3, 2]
    #x_lag = [3, 1] 
    #x_coeff = []
    #x_coeff = [0.5479120971119267, -0.12224312049589536, 0.7171958398227649, 2.7894721162374556]
    #x_coeff = [0.73, 0.62, -0.47, 0.23, -0.55]

    # causal coefficients to y
    y_start = list(np.random.randint(low=0, high=n_variables, size=n_xy))
    y_lag = list(np.random.randint(low=1, high=tau, size=n_xy))
    y_coeff = list(rng.uniform(-1,1,size=n_xy))

    for i in range(n_yy):
        y_start.append(n_variables)
        y_lag.append(np.random.randint(low=1, high=tau))
        y_coeff.append(rng.uniform(-1,1))

    #y_coeff = [-1.6232906084494019, 0.8512447032735118]
    #y_lag = [4, 3]
    #y_start = [2, 0, 5]
    #y_lag = [4, 2, 4]
    #y_coeff = [-0.9947384927474006, 0.0, 0.9696069504751108]

    ############ populate x,y ############

    # First tau values are randomly sampled
    x[0:tau,:] = rng.uniform(-1,1,size=(tau,n_variables))#np.random.randn(tau,n_variables)
    y[0:tau] = rng.uniform(-1,1,size=(tau))#np.random.randn(tau)

    # columns of x not having inputs are randomly sampled
    no_inputs = [k for k in set(range(n_variables)).difference(set(x_end))]
    if len(no_inputs)>0: 
        x[:,no_inputs] = rng.uniform(-1,1,size=(n_data,len(no_inputs)))

    #x = np.concatenate((x,y.reshape(-1,1)),axis=1)

    # Generate the rest of the values
    for i in range(n_data - tau):
        y[[tau+i]] = sum(x[[a - b for a, b in zip([tau+i]*n_xy, y_lag[0:n_xy])],y_start[0:n_xy]] * y_coeff[0:n_xy])
        y[[tau+i]] += sum(y[[a - b for a, b in zip([tau+i]*n_yy, y_lag[n_xy:])]] * y_coeff[n_xy:])
        #y[[tau+i]] += np.random.randn()*0.001
        #y[[tau+i]] -= (n_xy+n_yy)

        for j in range(n_variables):
            #bias = 0
            #compr = [a - b for a, b, c, d in zip(x_start,x_end,x_lag,x_coeff)]
            idx = [k for k in range(len(x_end)) if (x_end[k] == j) & (k<n_xx)]
            #bias = len(idx)
            x[[[tau+i]],j] = sum( x[[a - b for a, b in zip([tau+i]*len(idx), [x_lag[k] for k in idx])], [x_start[k] for k in idx]] * [x_coeff[k] for k in idx])
            idx = [k for k in range(len(x_end)) if (x_end[k] == j) & (k>=n_xx)]
            #bias += len(idx)
            x[[[tau+i]],j] += sum( y[[a - b for a, b in zip([tau+i]*len(idx), [x_lag[k] for k in idx])]] * [x_coeff[k] for k in idx])
            #x[[[tau+i]],j] += np.random.randn()*0.001
            #x[[[tau+i]],j] -= bias

    
    return x_start,x_end,x_lag,x_coeff,y_start,y_lag,y_coeff,x,y