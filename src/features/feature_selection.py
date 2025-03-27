#### We willl create one function that will do all the feature selection in one go and return the selected features data freame

from scipy.stats import chi2_contingency,f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from src.entities import FeatureParams
import numpy as np


def feature_selection_data(df_final_data_cust,target_column):
    # Chi-square test
    cat_columns_chi2 = []
    #Categorical columns
    categorical_cols = df_final_data_cust.select_dtypes(include=['object']).columns.tolist()
    # Remove 'Approved_Flag'
    if target_column in categorical_cols:
        categorical_cols.remove('Approved_Flag')
    for i in categorical_cols:
        chi2, pval, _, _ = chi2_contingency(pd.crosstab(df_final_data_cust[i], df_final_data_cust[target_column]))
        if pval <=0.05:
            cat_columns_chi2.append(i)

    # Apply sequential VIf for nyumerical columns
    # numerical columns list
    numeric_columns = []
    columns_to_be_kept = []
    for i in df_final_data_cust.columns:
        if df_final_data_cust[i].dtype != 'object' and i not in ['PROSPECTID',target_column]:
            numeric_columns.append(i)
    vif_data = df_final_data_cust[numeric_columns]
    total_columns = vif_data.shape[1]
    column_index = 0
    for i in range (0,total_columns):
        try:
            vif_value = variance_inflation_factor(vif_data, column_index)
            if np.isinf(vif_value) or np.isnan(vif_value):  
                vif_data = vif_data.drop([numeric_columns[i]], axis=1)
                continue
            if vif_value <= 6:
                columns_to_be_kept.append( numeric_columns[i] )
                column_index = column_index+1
            else:
                vif_data = vif_data.drop([ numeric_columns[i] ] , axis=1)
        except ZeroDivisionError:
            vif_data = vif_data.drop([numeric_columns[i]], axis=1)  
        
    # check Anova for columns_to_be_kept 
    from scipy.stats import f_oneway
    columns_to_be_kept_numerical = []
    for i in columns_to_be_kept:
        a = list(df_final_data_cust[i])  
        b = list(df_final_data_cust[target_column])     
        group_P1 = [value for value, group in zip(a, b) if group == 'P1']
        group_P2 = [value for value, group in zip(a, b) if group == 'P2']
        group_P3 = [value for value, group in zip(a, b) if group == 'P3']
        group_P4 = [value for value, group in zip(a, b) if group == 'P4']
        f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)
        if p_value <= 0.05:
            columns_to_be_kept_numerical.append(i)
    #Combine all
    selected_features_list_without_target=columns_to_be_kept_numerical + cat_columns_chi2
    #final_features_df=df_final_data_cust[features_list_without_target + ['Approved_Flag']]
    return selected_features_list_without_target