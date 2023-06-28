# Custom functions
import pandas as pd
import numpy as np
import os
from scipy import stats

from utils.config_reader import config_reader 
from sklearn.metrics import roc_curve, roc_auc_score


# Импортируем константы из файла config
config = config_reader('../config/config.json')
path_figures = config.path_figures





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

    
#     else:
#         return arg
    
    