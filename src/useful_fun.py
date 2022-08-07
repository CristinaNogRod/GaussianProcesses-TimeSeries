from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_distribution(X, Y, mean, var, pos, labels, legend=True):
    plt.figure(figsize=(25, 7))

    plt.plot(X, Y, '.', color="black", alpha=0.5, markersize=3, label='True (Test) Observation Samples')
    plt.plot(X, mean, color="C0", label='Mean Predictive Posterior')
    c = 1.96 * np.sqrt(var)
    plt.fill_between(X[:,0], (mean - c)[:,0], (mean + c)[:,0], alpha=0.2, edgecolor='gray', facecolor='C0', label='CI')

    plt.xticks(pos, labels)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Normalised Births', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=12)

    if legend:
        plt.legend()

    plt.show()
    plt.close()


def percentage_outof_CI(Y, mean, var):
    c = 1.96 * np.sqrt(var)
    lower = mean - c
    upper = mean + c
    points_outof_bounds = (Y < lower) | (Y > upper)
    return np.mean(points_outof_bounds) * 100