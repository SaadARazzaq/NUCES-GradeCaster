import pandas as pd
from pandas import ExcelWriter as Ew
import os

raw_data = "./data/Raw_Data/Dataset.xlsx"

'''
-> First we Normalize each sheet and then merge all the normalized sheets.
-> Benefit of Normalization: Normalization will help us to esily get insights from the data
|
 -> Benefit of Normalizing sheet by sheet first: This will help to get insights section wise as well as batch (i.e. 18, 19, 20, 21) wise.
'''

def normalize_raw_data(raw_data):
    data_sheets = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
    normalized_file_path = "./data/Normalized_Data/Dataset.xlsx"

    if os.path.exists(normalized_file_path):
        os.remove(normalized_file_path)  # Remove the file if it already exists

    xlsx_writer = Ew(normalized_file_path, mode='w')  # to write to a new Excel file

    '''
    Open Each sheet, iterate through it.
    Normalize it and read that sheet into the dataframe.
    Once the dataframe is normalized, Write it to the normalized_excel_file.
    '''

    for sheet in data_sheets:
            
            Assignments = ['As:1', 'As:2', 'As:3', 'As:4', 'As:5', 'As:6', 'As:7']
            
            Quizzes = ['Qz:1', 'Qz:2', 'Qz:3', 'Qz:4', 'Qz:5', 'Qz:6', 'Qz:7', 'Qz:8']

            df = pd.read_excel(raw_data, sheet_name= sheet)
            
            df.fillna(0, inplace=True)  # Impute the empty cols with 0
            
            # Proportional Scaling Normalization

            for column in df.columns:
                # Check if the column belongs to 'Assignments'
                if column in Assignments:
                    # Iterate over the rows starting from the fourth row
                    for i in range(3, len(df[column])):
                        # Calculate and update each value in the column
                        df.loc[i, column] = (df.loc[i, column] / df.loc[1, column]) * (df.loc[0, column] / 15)

                # Check if the column belongs to 'Quizzes'
                if column in Quizzes:
                    # Iterate over the rows starting from the fourth row
                    for i in range(3, len(df[column])):
                        # Calculate and update each value in the column
                        df.loc[i, column] = (df.loc[i, column] / df.loc[1, column]) * (df.loc[0, column] / 10)

            # Write the normalized DataFrame to the new Excel file
            df.to_excel(xlsx_writer, sheet_name=sheet, index=False)
    
    # Save the Excel file
    xlsx_writer.close()