# Custom functions
import pandas as pd
import numpy as np
import os
import scipy

from utils.config_reader import config_reader 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# Импортируем константы из файла config
config = config_reader('../config/config.json')
path_figures = config.path_figures



def get_ROC_plot(model,  X_test, y_test, title:str,  plot_counter:int=None):
    """
    Create the roc curve plot
    Args:
        model (_type_): pre-trained model to get prediction
        X_test (_type_): X matrice with 
        y_test (_type_): y predicted values
        title (_type_): _description_
        figpath (_int_): figure path for saving
    """
    y_pred = model.predict_proba(X_test)[:,1]
    metric = roc_auc_score(y_test, y_pred).round(3)
    print('roc_auc: ', metric)

    false_positive_rates, true_positive_rates, threshold = roc_curve(y_test, y_pred)


    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))

    # ROC curve
    ax.plot(false_positive_rates, true_positive_rates, 
            label='Smoothed values ROC-AUC')

    # Random model
    ax.plot([0, 1], [0, 1], color='k', lw=2, linestyle=':', 
            label='Model predicting random')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    ax.fill_between(false_positive_rates, 
                    true_positive_rates, 
                    step="pre", 
                    alpha=0.4, label='Area under curve (ROC-AUC)')
    
    # Annotate figure with ROC curve
    plt.annotate(f'ROC: {metric}', 
        xy=(0.45,0.6), textcoords='data', 
        bbox={'facecolor': 'w', 'alpha': 0.95, 'pad': 10} 
    );
    
    ax.legend()
    
    
    if plot_counter is not None:
        ax.set_title(f'Fig.{plot_counter} - ROC curve for {title}', y=-0.25,fontsize=13, loc='center')
        plt.tight_layout()
        plt.savefig(path_figures + f'fig_{plot_counter}.png')
        
    else:
        #plot_counter=1
        plt.tight_layout()
        ax.set_title(f'ROC curve for {title}', y=-0.25)
        
        
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
        



# def get_purpose(arg:str)->str:
#     """Categorization of loan needs

#     Args:
#         arg (str): need description

#     Returns:
#         _type_: categorized need
#     """    
    
#     arg = arg.lower().replace('/',' ')
    
#     debt_consolidation = ['debt_consolidation']
#     cred_card = ['credit_card']
#     house = ['home_improvement', 'house','moving','renewable_energy']
#     purchases = ['car', 'major_purchase']
#     education = ['educational']
#     business = ['small_business','business']
#     health = ['medical']
#     leasure = ['vacation']
    
#     if arg in debt_consolidation or ('debt' in arg  and 'consolidation' in arg): 
#         return 'debt_consolidation' 
    
#     elif arg in cred_card:
#         return 'cred_card'

#     elif arg in house:
#         return 'house'    

#     elif arg in purchases:
#         return 'purchases' 
    
#     elif arg in education:
#         return 'education' 
    
#     elif arg in business:
#         return 'business' 
    
#     elif arg in health:
#         return 'health'     
        
#     elif arg in leasure:
#         return 'leasure'  
    
#     else:
#         return arg
    
    
def annotate_scatterplot(fig)->dict:
    """display r2 and fit equation  at the scatterplot

    Args:
        fig (_type_): figure instance

    Returns:
        _dict_: slope, intercept and rvalue for every line
    """    
    args = dict()
    
    for i in range(len(fig.get_lines())):
        args[i] = list(scipy.stats.linregress(x=fig.get_lines()[i].get_xdata(),y=fig.get_lines()[i].get_ydata()))[:3]
    
    return args