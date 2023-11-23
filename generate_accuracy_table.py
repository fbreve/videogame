# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:05:31 2023

@author: fbrev
"""

import glob
import os
import numpy as np
import csv

platforms = [28, 22, 26, 40, 16, 15, 23, 6, 9, 8, 7, 14, 13, 69, 81, 82, 132, 141, 142, 203, 289, 288]

res_table = np.empty((0, 4))

def is_valid_row(row):
    return len(row) == expected_column_count
   
for platform in platforms:
    #print(platform)
    # Define path to results
    path = "results/"
    # Define the pattern to match the results file
    file_pattern = 'vg-platform' + str(platform)  + '-*.csv'
    # Use glob to search for files matching the pattern in the directory
    matching_files = glob.glob(os.path.join(path, file_pattern))
    # Check if any matching files were found
    if len(matching_files) == 0:
        print("Results file not found.")
        exit(0)

    with open(matching_files[0], 'r') as f_csv:
        csv_reader = csv.reader(f_csv, delimiter=',')
        # Initialize an empty list to store valid rows
        valid_rows = []    
        # Define the expected number of columns in a valid row
        expected_column_count = 5
        
        # Iterate through the rows in the CSV file
        for row in csv_reader:
            if is_valid_row(row):
                valid_rows.append(row)
            
    acc_tab = np.array(valid_rows, dtype=float)      
    
    # get only the first four rows
    acc_tab = acc_tab[:4]
    
    # Calculate the mean for each row
    means = np.mean(acc_tab, axis=1)

    res_table = np.vstack((res_table, means))

csv_results_file = 'results/results_table.csv'  
np.savetxt(csv_results_file, res_table, delimiter=',', fmt='%.4f')