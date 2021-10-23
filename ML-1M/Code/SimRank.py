import torch
import numpy as np
import pandas as pd
import pickle as pkl
import config
from train import load_model

args = config.args_initialization()


def normalization(row):
    val = 0
    for i in range(len(row)):
        if row[i] > 0:
            val += 1
    return row / val


def simrank(args):
    user_index_to_uid = pkl.load(open(args.path + 'user_index_to_uid.pkl', 'rb'))
    movie_index_to_mid = pkl.load(open(args.path + 'movie_index_to_mid.pkl', 'rb'))
    R = pkl.load(open(args.path + 'rank_matrix_initial.pkl', 'rb'))
    C = pkl.load(open(args.path + 'choice_matrix.pkl', 'rb'))  # Choice Matrix
    C_col = C.apply(normalization, axis=0)  # 列归一化
    C_row = C.apply(normalization, axis=1)  # 行归一化
    X = pd.DataFrame(np.zeros([6040, 6040], dtype=int))
    Y = pd.DataFrame(np.zeros([3706, 3706], dtype=int))
    
    for k in range(args.k):
        X_new = args.c * np.dot(np.dot(C_row, Y), np.matrix(C_row).T) + np.eye(6040) - np.diag(
            np.diagonal(args.c * np.dot(np.dot(C_row, Y), np.matrix(C_row).T)))
        Y_new = args.c * np.dot(np.dot(C_col.T, X), C_col) + np.eye(3706) - np.diag(
            np.diagonal(args.c * np.dot(np.dot(C_col.T, X), C_col)))
        X = X_new
        Y = Y_new
    
    def series_notzero_values(series):
        val = 0
        for i in series.index:
            if series[i] > 0:
                val += 1
        return val
    
    for i in R.index:
        for j in R.columns:
            if R.iloc[i][j] > 0:
                pass
            else:
                val = 0
                for p in R.columns:
                    val += C.iloc[i][p] * R.iloc[i][p] * Y[p, j]
                for q in R.index:
                    val += C.iloc[q][j] * R.iloc[q][j] * X[q, i]
                not_zero_val = series_notzero_values(C[j]) + series_notzero_values(C.iloc[i])
                # not_zero_val = series_notzero_values(C.iloc[i])
                R[j][i] = val / not_zero_val
    
    return R
