import pandas as pd
import scipy.stats as stats
import statistics

# Get average predicted intensity
# df = pd.read_csv('predicted_results_eec_babbage_with_prob.csv')
df = pd.read_csv('calibrate/calibrated_b_mean_results.csv')

# group_by_all_df = df.groupby(['emotion', 'emotion_word', 'race', 'template', 'gender']).agg(
#     avg_intensity=pd.NamedAgg(column='predicted_intensity', aggfunc=statistics.mean)
# )

group_by_name_df = df.groupby(['emotion', 'emotion_word', 'race', 'template', 'gender']).agg(
    avg_intensity=pd.NamedAgg(column='calibrated_intensity', aggfunc=statistics.mean)
)

group_by_non_name_df = df[pd.isnull(df["race"])].groupby(['emotion', 'emotion_word', 'template', 'gender']).agg(
    avg_intensity=pd.NamedAgg(column='calibrated_intensity', aggfunc=statistics.mean)
)

group_by_name_df.to_csv('analysis/babbage/group_by_name_table_calibrated_raw.csv')
group_by_non_name_df.to_csv('analysis/babbage/group_by_non_name_table_calibrated_raw.csv')




# Get ttest result for gender bias

group_by_all_df = pd.read_csv('analysis/babbage/group_by_name_table_calibrated_raw.csv')

pivoted_group_by_all_df = group_by_all_df.pivot(columns='gender', values='avg_intensity')
female_predictions = pivoted_group_by_all_df.female.values.tolist()
female_predictions = [x for x in female_predictions if str(x) != 'nan']
male_predictions = pivoted_group_by_all_df.male.values.tolist()
male_predictions = [x for x in male_predictions if str(x) != 'nan']

diff = []
for idx, pred in enumerate(female_predictions):
    diff.append(pred - male_predictions[idx])

se = stats.sem(diff)

print('standard error --->', se)

result = stats.ttest_rel(female_predictions, male_predictions)
print('number of pairs --->', len(male_predictions))
print('----->ttest result for gender bias: ', result)


group_by_all_df = pd.read_csv('analysis/babbage/group_by_non_name_table_calibrated_raw.csv')

pivoted_group_by_all_df = group_by_all_df.pivot(columns='gender', values='avg_intensity')
female_predictions = pivoted_group_by_all_df.female.values.tolist()
female_predictions = [x for x in female_predictions if str(x) != 'nan']
male_predictions = pivoted_group_by_all_df.male.values.tolist()
male_predictions = [x for x in male_predictions if str(x) != 'nan']

diff = []
for idx, pred in enumerate(female_predictions):
    diff.append(pred - male_predictions[idx])

se = stats.sem(diff)

print('standard error --->', se)

result = stats.ttest_rel(female_predictions, male_predictions)
print('number of pairs --->', len(male_predictions))
print('----->ttest result for gender bias: ', result)


# Get ttest result for race bias
group_by_all_df = pd.read_csv('analysis/babbage/group_by_name_table_calibrated_raw.csv')

pivoted_group_by_all_df = group_by_all_df.pivot(columns='race', values='avg_intensity')
euro_predictions = pivoted_group_by_all_df['European'].values.tolist()
euro_predictions = [x for x in euro_predictions if str(x) != 'nan']
aa_predictions = pivoted_group_by_all_df['African-American'].values.tolist()
aa_predictions = [x for x in aa_predictions if str(x) != 'nan']

diff = []
for idx, pred in enumerate(euro_predictions):
    diff.append(pred - aa_predictions[idx])

se = stats.sem(diff)

print('standard error --->', se)

result = stats.ttest_rel(euro_predictions, aa_predictions)
print('number of pairs --->', len(aa_predictions))
print('----->ttest result for race bias: ', result)


# Baseline
# standard error ---> 0.01725847468436841
# number of pairs ---> 280
# ----->ttest result for gender bias:  TtestResult(statistic=-3.2696152157456146, pvalue=0.001212017487193234, df=279)
# standard error ---> 0.032399908553753465
# number of pairs ---> 140
# ----->ttest result for gender bias:  TtestResult(statistic=-4.519413108393109, pvalue=1.3145410504337435e-05, df=139)
# standard error ---> 0.009002636988823991
# number of pairs ---> 280
# ----->ttest result for race bias:  TtestResult(statistic=-3.17367329226954, pvalue=0.001673584987359671, df=279)

# W multiply
# standard error ---> 0.037404548991738965
# number of pairs ---> 280
# ----->ttest result for gender bias:  TtestResult(statistic=9.280765749094472, pvalue=4.8690455399701894e-18, df=279)
# standard error ---> 0.04956424417255263
# number of pairs ---> 140
# ----->ttest result for gender bias:  TtestResult(statistic=7.493881479065482, pvalue=7.086882859970086e-12, df=139)
# standard error ---> 0.04031221869640828
# number of pairs ---> 280
# ----->ttest result for race bias:  TtestResult(statistic=9.78079914814622, pvalue=1.2868428278159334e-19, df=279)

# W mean
# standard error ---> 0.0104622166264651
# number of pairs ---> 280
# ----->ttest result for gender bias:  TtestResult(statistic=3.277098487808554, pvalue=0.0011815152946658724, df=279)
# standard error ---> 0.04956424417255263
# number of pairs ---> 140
# ----->ttest result for gender bias:  TtestResult(statistic=7.493881479065482, pvalue=7.086882859970086e-12, df=139)
# standard error ---> 0.021267701323749718
# number of pairs ---> 280
# ----->ttest result for race bias:  TtestResult(statistic=5.911042467664342, pvalue=9.897652700110067e-09, df=279)

# b
# standard error ---> 0.03094821548244425
# number of pairs ---> 280
# ----->ttest result for gender bias:  TtestResult(statistic=8.793491091505208, pvalue=1.5284371182453648e-16, df=279)
# standard error ---> 0.04559014490635654
# number of pairs ---> 140
# ----->ttest result for gender bias:  TtestResult(statistic=7.081731008319907, pvalue=6.472906720665125e-11, df=139)
# standard error ---> 0.040191348881404035
# number of pairs ---> 280
# ----->ttest result for race bias:  TtestResult(statistic=10.396693678785171, pvalue=1.2992210937870004e-21, df=279)

# b mean
# standard error ---> 0.009862543694316526
# number of pairs ---> 280
# ----->ttest result for gender bias:  TtestResult(statistic=1.2312094647402876, pvalue=0.21928136906802934, df=279)
# standard error ---> 0.04559014490635654
# number of pairs ---> 140
# ----->ttest result for gender bias:  TtestResult(statistic=7.081731008319907, pvalue=6.472906720665125e-11, df=139)
# standard error ---> 0.021407183386625308
# number of pairs ---> 280
# ----->ttest result for race bias:  TtestResult(statistic=5.638961629567068, pvalue=4.1902783312088014e-08, df=279)
