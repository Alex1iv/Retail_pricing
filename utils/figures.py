# Custom functions
import pandas as pd
import numpy as np
import os
import seaborn as sns
from scipy import stats

from utils.config_reader import config_reader 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# Импортируем константы из файла config
config = config_reader('../config/config.json')
path_figures = config.path_figures

def qq_plot(data:pd.DataFrame, features:list, target:str):
    # display pair plots
    for i in data[features]:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
        
        sns.histplot(data[i], kde=True, bins=30, ax=ax[0]) # histogram
        ax[0].set_title('Histogram')

        stats.probplot(data[i], plot=ax[1]) # qq plot
        
        # scatter plot with regression
        regplot = sns.regplot(data=data, x=i, y=target, ax=ax[2], line_kws={"color": "red"}) #, scatter_kws={"color": "blue"},
        regplot.set_title(f'{i} VS {target}')
        regplot.xaxis.set_tick_params(rotation=45)
                
        
        plt.tight_layout()
        fig.suptitle(f'Feature {i} distribution and comparison with the target \n------------------', y=-0.05);
        


        
        
def get_comparison(models:dict, X_test, y_test, path_figures=config.path_figures, fig_id:int=None): #title:str, 
    """
    Plot roc curves of given models from dictionary.
    Args:
        model (_dict_): pre-trained models dictionary
        X_test (_type_): X matrice with test values
        y_test (_type_): y predicted values
        figpath (_int_): figure path for saving
    """
    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))

    # Random model
    ax.plot([0, 1], [0, 1], color='k', lw=2, linestyle=':', 
            label='Random classifier')

    for name, model in models.items():
        y_pred = model.predict_proba(X_test)[:,1]
        metric = roc_auc_score(y_test, y_pred).round(3)
        
        false_positive_rates, true_positive_rates, threshold = roc_curve(y_test, y_pred)

        # ROC curve
        ax.plot(false_positive_rates, true_positive_rates, label=f'{name}')

    ax.set_title(f'Fig.{fig_id} - Models comparison by ROC curves', y=-0.2) 
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    ax.legend()
    plt.tight_layout();
    
    if fig_id:
        plt.savefig(os.path.join(path_figures + f'fig_{fig_id}.png'))
        


    
    
def annotate_scatterplot(fig)->dict:
    """display r2 and fit equation  at the scatterplot

    Args:
        fig (_type_): figure instance

    Returns:
        _dict_: slope, intercept and rvalue for every line
    """    
    args = dict()
    
    for i in range(len(fig.get_lines())):
        args[i] = list(stats.linregress(x=fig.get_lines()[i].get_xdata(),y=fig.get_lines()[i].get_ydata()))[:3]
    
    return args