import pandas as pd
from pandas import ExcelWriter as Ew
import os

normalized_data = "./data/Normalized_Data/Dataset.xlsx"

def merger(normalized_data):

    data_sheets = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
    merged_file_path = "./data/Normalized_Merged_Data/Dataset.xlsx"

    merged_df = []  # An empty list to store DataFrames of each sheet
    
    for sheet in data_sheets:  # Iterates in each sheet then reads in the dataframe

        df = pd.read_excel(normalized_data, sheet_name=sheet, skiprows=[1,2,3])
        merged_df.append(df)

    df = pd.concat(merged_df, ignore_index=True)  #  Append all the sheets to one dataframe

    if os.path.exists(merged_file_path):
        os.remove(merged_file_path)  # Remove the file if it already exists

    xlsx_writer = Ew(merged_file_path, mode='w')

    df.fillna(0, inplace=True)
    
    df.to_excel(xlsx_writer, index=False)

    xlsx_writer.close()