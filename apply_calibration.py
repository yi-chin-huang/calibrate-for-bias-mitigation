import pandas as pd
import numpy as np
import statistics

calibration_input = pd.read_csv("calibrate/calibration_input.csv")

df = pd.read_csv('predicted_results_eec_babbage_with_prob.csv')

# adjust predicted_prob_one_exp columns (1 -> 0)
df['predicted_prob_one_factor'] = np.where(df['predicted_prob_one_exp'] == 1, 0, 1)
df['predicted_prob_one_exp'] = df['predicted_prob_one_exp'] * df['predicted_prob_one_factor']

# apply calibration
df['1'] = df['predicted_prob_one_exp'] * cp_one
df['2'] = df['predicted_prob_two_exp'] * cp_two
df['3'] = df['predicted_prob_three_exp'] * cp_three
df['4'] = df['predicted_prob_four_exp'] * cp_four
df['5'] = df['predicted_prob_five_exp'] * cp_five

# find calibrated output
df['calibrated_intensity'] = df.iloc[:,-5:].idxmax(axis=1)
df['calibrated_intensity'] = df['calibrated_intensity'].astype(int)

# output to csv
df.to_csv('calibrate/'+str(cp_one)+"_"+str(cp_two)+"_"+str(cp_three)+"_"+str(cp_four)+"_"+str(cp_five)+".csv")

# calculate mean
