import pandas as pd
import numpy as np
import statistics

calibration_input = pd.read_csv("calibrate/calibrate_input_raw.csv")

# calculate by-group averages
calibration_input = calibration_input.groupby(['race', 'gender', 'emotion']).agg(
    prob_one=pd.NamedAgg(column='predicted_prob_one_exp', aggfunc=statistics.mean),
    prob_two=pd.NamedAgg(column='predicted_prob_two_exp', aggfunc=statistics.mean),
    prob_three=pd.NamedAgg(column='predicted_prob_three_exp', aggfunc=statistics.mean),
    prob_four=pd.NamedAgg(column='predicted_prob_four_exp', aggfunc=statistics.mean),
    prob_five=pd.NamedAgg(column='predicted_prob_five_exp', aggfunc=statistics.mean)
).reset_index()

# extract neutral values and rename columns
neutral_input = calibration_input.loc[calibration_input['gender'] == 'neutral']
neutral_input = neutral_input.rename(columns={'prob_one': 'prob_one_neutral', 
                              'prob_two': 'prob_two_neutral', 
                              'prob_three': 'prob_three_neutral', 
                              'prob_four': 'prob_four_neutral', 
                              'prob_five': 'prob_five_neutral'})
neutral_input = neutral_input.drop(columns=['gender', 'race'])
# neutral_input.to_csv('calibrate/test1.csv')

# filter out neutral values
calibration_input = calibration_input.loc[calibration_input['gender'] != 'neutral']

# create gender table
gender_table = calibration_input.loc[calibration_input['gender'] != 'None']
gender_table = gender_table.drop(columns='race')
gender_table = gender_table.merge(neutral_input, how = 'left', on = ['emotion'])
gender_table['gender_calibrate_1'] = gender_table['prob_one_neutral'] / gender_table['prob_one']
gender_table['gender_calibrate_2'] = gender_table['prob_two_neutral'] / gender_table['prob_two']
gender_table['gender_calibrate_3'] = gender_table['prob_three_neutral'] / gender_table['prob_three']
gender_table['gender_calibrate_4'] = gender_table['prob_four_neutral'] / gender_table['prob_four']
gender_table['gender_calibrate_5'] = gender_table['prob_five_neutral'] / gender_table['prob_five']
gender_calibrate_cols = ['gender_calibrate_1', 'gender_calibrate_2', 'gender_calibrate_3', 'gender_calibrate_4', 'gender_calibrate_5']
gender_table['gender_calibrate_1_normalized'] = gender_table['gender_calibrate_1'] / gender_table[gender_calibrate_cols].mean(axis = 1)
gender_table['gender_calibrate_2_normalized'] = gender_table['gender_calibrate_2'] / gender_table[gender_calibrate_cols].mean(axis = 1)
gender_table['gender_calibrate_3_normalized'] = gender_table['gender_calibrate_3'] / gender_table[gender_calibrate_cols].mean(axis = 1)
gender_table['gender_calibrate_4_normalized'] = gender_table['gender_calibrate_4'] / gender_table[gender_calibrate_cols].mean(axis = 1)
gender_table['gender_calibrate_5_normalized'] = gender_table['gender_calibrate_5'] / gender_table[gender_calibrate_cols].mean(axis = 1)
gender_table = gender_table[['gender', 'emotion', 'gender_calibrate_1_normalized', 'gender_calibrate_2_normalized', 'gender_calibrate_3_normalized', 'gender_calibrate_4_normalized', 'gender_calibrate_5_normalized']]

# create race table
race_table = calibration_input.loc[calibration_input['race'] != 'None']
race_table = race_table.drop(columns='gender')
race_table = race_table.merge(neutral_input, how = 'left', on = ['emotion'])
race_table['race_calibrate_1'] = race_table['prob_one_neutral'] / race_table['prob_one']
race_table['race_calibrate_2'] = race_table['prob_two_neutral'] / race_table['prob_two']
race_table['race_calibrate_3'] = race_table['prob_three_neutral'] / race_table['prob_three']
race_table['race_calibrate_4'] = race_table['prob_four_neutral'] / race_table['prob_four']
race_table['race_calibrate_5'] = race_table['prob_five_neutral'] / race_table['prob_five']
race_calibrate_cols = ['race_calibrate_1', 'race_calibrate_2', 'race_calibrate_3', 'race_calibrate_4', 'race_calibrate_5']
race_table['race_calibrate_1_normalized'] = race_table['race_calibrate_1'] / race_table[race_calibrate_cols].mean(axis = 1)
race_table['race_calibrate_2_normalized'] = race_table['race_calibrate_2'] / race_table[race_calibrate_cols].mean(axis = 1)
race_table['race_calibrate_3_normalized'] = race_table['race_calibrate_3'] / race_table[race_calibrate_cols].mean(axis = 1)
race_table['race_calibrate_4_normalized'] = race_table['race_calibrate_4'] / race_table[race_calibrate_cols].mean(axis = 1)
race_table['race_calibrate_5_normalized'] = race_table['race_calibrate_5'] / race_table[race_calibrate_cols].mean(axis = 1)
race_table = race_table[['race', 'emotion', 'race_calibrate_1_normalized', 'race_calibrate_2_normalized', 'race_calibrate_3_normalized', 'race_calibrate_4_normalized', 'race_calibrate_5_normalized']]

# read in main table and drop unnecessary columns
df = pd.read_csv('predicted_results_eec_babbage_with_prob.csv')
df = df.drop(columns=['Unnamed: 0', 'predicted_prob_one', 'predicted_prob_two', 'predicted_prob_three', 'predicted_prob_four', 'predicted_prob_five'])

# adjust predicted_prob_one_exp columns (1 -> 0)
df['predicted_prob_one_factor'] = np.where(df['predicted_prob_one_exp'] == 1, 0, 1)
df['predicted_prob_one_exp'] = df['predicted_prob_one_exp'] * df['predicted_prob_one_factor']
df = df.drop(columns='predicted_prob_one_factor')

# merge in calibration parameters
df = df.merge(gender_table, how = 'left', on = ['gender', 'emotion'])
df = df.merge(race_table, how = 'left', on = ['race', 'emotion'])
df['1'] = df['predicted_prob_one_exp'] * (df['gender_calibrate_1_normalized'] + df['race_calibrate_1_normalized']) / 2
df['2'] = df['predicted_prob_two_exp'] * (df['gender_calibrate_2_normalized'] + df['race_calibrate_2_normalized']) / 2
df['3'] = df['predicted_prob_three_exp'] * (df['gender_calibrate_3_normalized'] + df['race_calibrate_3_normalized']) / 2
df['4'] = df['predicted_prob_four_exp'] * (df['gender_calibrate_4_normalized'] + df['race_calibrate_4_normalized']) / 2
df['5'] = df['predicted_prob_five_exp'] * (df['gender_calibrate_5_normalized'] + df['race_calibrate_5_normalized']) / 2
df.loc[df['race_calibrate_1_normalized'].isnull(), '1'] = df['predicted_prob_one_exp'] * df['gender_calibrate_1_normalized']
df.loc[df['race_calibrate_2_normalized'].isnull(), '2'] = df['predicted_prob_two_exp'] * df['gender_calibrate_2_normalized']
df.loc[df['race_calibrate_3_normalized'].isnull(), '3'] = df['predicted_prob_three_exp'] * df['gender_calibrate_3_normalized']
df.loc[df['race_calibrate_4_normalized'].isnull(), '4'] = df['predicted_prob_four_exp'] * df['gender_calibrate_4_normalized']
df.loc[df['race_calibrate_5_normalized'].isnull(), '5'] = df['predicted_prob_five_exp'] * df['gender_calibrate_5_normalized']

# find calibrated output
df['calibrated_intensity'] = df.iloc[:,-5:].idxmax(axis=1)
df['calibrated_intensity'] = df['calibrated_intensity'].astype(int)
df.to_csv("calibrate/calibrated_mean_results.csv")

# test = df.groupby(['predicted_intensity', 'calibrated_intensity']).count()
# test.to_csv("calibrate/test6.csv")

# # output to csv
# df.to_csv('calibrate/'+str(cp_one)+"_"+str(cp_two)+"_"+str(cp_three)+"_"+str(cp_four)+"_"+str(cp_five)+".csv")

# # calculate mean
