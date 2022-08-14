from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_distribution(X, Y, mean, var, pos, labels, method='bayes', Z=None):
    plt.figure(figsize=(25, 7))

    plt.plot(X, Y, '.', color="black", alpha=0.5, markersize=3, label='True (Test) Observation Samples')
    plt.plot(X, mean, color="C0", label='Mean Predictive Posterior')
    c = 1.96 * np.sqrt(var)
    if method == 'freq':
        plt.fill_between(X[:,0], (mean - c)[:,0], (mean + c)[:,0], alpha=0.2, edgecolor='gray', facecolor='C0', label='CI')
    if method == 'bayes':
        #Z = m.inducing_variable.Z.numpy()
        plt.plot(Z, np.zeros_like(Z), "k|", color='grey', mew=2, label="Inducing Locations")
        plt.fill_between(X[:,0], (mean - c)[:,0], (mean + c)[:,0], alpha=0.2, edgecolor='gray', facecolor='C0', label='CI')

    plt.xticks(pos, labels)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Normalised Births', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(prop={'size':14})
    plt.show()
    plt.close()


def percentage_outof_CI(Y, mean, var):
    c = 1.96 * np.sqrt(var)
    lower = mean - c
    upper = mean + c
    points_outof_bounds = (Y < lower) | (Y > upper)
    return np.mean(points_outof_bounds) * 100


def number_outof_CI(Y, mean, var):
    c = 1.96 * np.sqrt(var)
    lower = mean - c
    upper = mean + c
    points_outof_bounds = (Y < lower) | (Y > upper)
    return np.sum(points_outof_bounds) 


def plot_sliding_window(x_train, x_test, y_train, y_test, mean_train, mean_test, var_train, var_test, pos, labels, iteration):
    plt.figure(figsize=(20,7))
    plt.xticks(pos, labels, rotation=45)
    plt.plot(x_train, y_train, '.', label='Train data', c='black', markersize=3)
    plt.plot(x_test, y_test, 'x', label='Test data', c='red', markersize=5)

    plt.plot(x_train, mean_train, '-', label='Mean Posterior for train data', c='C0', linewidth=3)
    c_train = 1.96 * np.sqrt(var_train) 
    plt.fill_between(x_train[:,0], (mean_train - c_train)[:,0], (mean_train + c_train)[:,0], alpha=0.4, edgecolor='gray', facecolor='C0', label='CI for train')

    plt.vlines(x_test[0], colors='grey', linestyles='dashed', ymin=np.min(y_train), ymax=np.max(y_train))

    c_test = 1.96 * np.sqrt(var_test) 
    plt.plot(x_test, mean_test, '-', label='Mean Predictive Posterior for test data', c='maroon', linewidth=3)
    plt.fill_between(x_test[:,0], (mean_test - c_test)[:,0], (mean_test + c_test)[:,0], alpha=0.4, edgecolor='gray', facecolor='grey', label='CI for test')
    
    plt.title('Sliding window for iteration ' + str(iteration))
    plt.xlabel('Date')
    plt.ylabel('Normalised number of births')
    
    plt.legend()
    plt.show()


def split_dataframe_by_position(df, splits):
    """
    Takes a dataframe and an integer of the number of splits to create.
    Returns a list of dataframes.
    """
    dataframes = []
    index_to_split = len(df) // splits
    start = 0
    end = index_to_split
    for split in range(splits):
        temporary_df = df.iloc[start:end, :]
        dataframes.append(temporary_df)
        start += index_to_split
        end += index_to_split
    return dataframes