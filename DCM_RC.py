#!/usr/bin/env python
# coding: utf-8
import time

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, Product, ConstantKernel
from sklearn.gaussian_process.kernels import RBF
shift_k = 0
approx_res_size = 1000

model_params = {'tau': 0.02,
                'nstep': 1000,
                'N': 3,
                'd': 3}

res_params = {'radius':0.2,
             'degree': 100,
             'sigma': 0.8,
            'Dr': 1000,
             'train_length': 1000,
             'N': int(np.floor(approx_res_size/model_params['N']) * model_params['N']),
             'num_inputs': model_params['N'],
             'predict_length': 1000,
             'beta': 0.0000010
              }

def generate_reservoir(size,radius,degree):
    sparsity = degree/float(size);
    A = sparse.rand(size,size,density=sparsity).todense()
    vals = np.linalg.eigvals(A)
    e = np.max(np.abs(vals))
    A = (A/e) * radius
    return A

def reservoir_layer(A, Win, input, res_params):
    states = np.zeros((res_params['train_length'],res_params['Dr']))
    states[0] = np.zeros(res_params['Dr'])
    for i in range(1, res_params['train_length']):
        states[i] = np.tanh(A.dot(states[i-1]) + Win.dot(input[i-1]) )
    states_nearfuture = np.tanh(A.dot(states[res_params['train_length']-1]) +Win.dot(input[res_params['train_length']-1]) )
    return states,states_nearfuture

def train_reservoir(res_params, data):
    A = generate_reservoir(res_params['Dr'], res_params['radius'], res_params['degree'])
    q = int(res_params['Dr']/res_params['num_inputs'])
    Win = np.zeros((res_params['Dr'],res_params['num_inputs']))
    for i in range(res_params['num_inputs']):
        np.random.seed(seed=i)
        Win[i*q: (i+1)*q,i] = res_params['sigma'] * (-1 + 2 * np.random.rand(1,q)[0])
    states,states_nearfuture = reservoir_layer(A, Win, data, res_params)
    Wout = train(res_params, states, data)
    return states_nearfuture, Wout, A, Win

def train(res_params,states,data):
    beta = res_params['beta']
    RC_states = np.hstack( (states, np.power(states, 2), np.power(states, 3) ,np.power(states, 4)) )
    # assert RC_states.shape == (res_params['train_length'], 5*res_params['Dr']), 'xtrain.shape error'
    Wout = np.linalg.inv(RC_states.T.dot(RC_states) + beta * np.eye(4 * res_params['Dr'])).dot(RC_states.T).dot(data[0:res_params['train_length']])
    return Wout

def predict(A, Win, res_params, states_nearfuture, Wout,gp_predictions):
    states = np.zeros((res_params['predict_length'], res_params['Dr']))
    output_states = np.zeros((res_params['predict_length'], 4*res_params['Dr']))
    states[0] = states_nearfuture
    output_states[0] = np.hstack( (states[0], np.power(states[0], 2),np.power(states[0], 3),np.power(states[0], 4)) )
    output = np.zeros((res_params['num_inputs'], res_params['predict_length']))
    gp_predictions_x = gp_predictions[:, 2:3]
    length = len(gp_predictions_x)
    j = 1
    for i in range(1,res_params['predict_length']):
        if j < length:
            out = output_states[i-1].dot(Wout)
            out[:1] = gp_predictions_x[j - 1:j]
            output[:, i-1] = out
            j = j + 1
        else:
            out = output_states[i-1].dot(Wout)
            out[:1] = gp_predictions_x[j - 1:j]
            output[:, i-1] = out
            gp_output = output[: ,i - length :i ]
            gp_predictions_next = gpr.predict(np.transpose(gp_output))
            gp_predictions_x = gp_predictions_next[:, 2:3]
            j = 1

        states[i] = np.tanh( A.dot(states[i-1]) + Win.dot( out ) )
        output_states[i] = np.hstack( (states[i], np.power(states[i], 2),np.power(states[i], 3),np.power(states[i], 4)) )

    predict = output_states.dot(Wout)
    return predict


data = np.load('lorenz_63.npy')

states_nearfuture,Wout,A,Win = train_reservoir(res_params,data[:,shift_k:shift_k+res_params['train_length']])
data_delay = np.transpose(data[:res_params['train_length']])
xs = data_delay[0, :]

tau_min = 5
data_lag0 = xs
data_lag1 = np.roll(xs, -tau_min)
data_lag2 = np.roll(xs, -2 * tau_min)
delay_data_x = np.vstack((data_lag0, data_lag1, data_lag2))
delay_data_x = np.transpose(delay_data_x)
train_data = data[:res_params['train_length']-(2 * tau_min)]
test_data = data[res_params['train_length']-(2 * tau_min):res_params['train_length'] ]

train_X = train_data
train_Y = delay_data_x[:res_params['train_length']-(2 * tau_min)]
test_X = test_data
kernell = RBF(length_scale=0.1, length_scale_bounds=(0.1, 10.0))  
gpr = GaussianProcessRegressor(kernel=kernell,
              normalize_y=True,
              n_restarts_optimizer=3,  # number of random starts to find the gaussian process hyperparameters
              random_state=10)

gpr.fit(train_X, train_Y)
gp_predictions = gpr.predict(test_X)

output = predict(A, Win,res_params,states_nearfuture,Wout,gp_predictions)


