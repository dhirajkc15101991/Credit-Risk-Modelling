
#create the function to remove df_internal_data_cust
def null_removal_df_internal_data_cust(internal_data_cust):
    df_internal_data_cust_final=internal_data_cust.loc[internal_data_cust['Age_Oldest_TL'] != -99999]
    return df_internal_data_cust_final

#create function for null removal of df_cibil_data_cust

def null_removal_df_cibil_data_cust(cibil_data_cust):
    threshold = 20 
    # Loop through all columns and drop those with >20% -99999 values
    columns_to_drop = []
    for col in cibil_data_cust.columns:
        null_percentage = (cibil_data_cust[col] == -99999).sum() / len(cibil_data_cust) * 100
        if null_percentage > threshold:
            columns_to_drop.append(col)
    # Drop the identified columns
    cibil_data_cust.drop(columns=columns_to_drop, inplace=True)
    # Step 2: Drop rows where any remaining column contains -99999
    df_cibil_data_cust_final = cibil_data_cust[cibil_data_cust.ne(-99999).all(axis=1)]
    return df_cibil_data_cust_final