import pandas as pd
import scipy.stats as stats
import statistics

# Get average predicted intensity
# df = pd.read_csv('predicted_results_eec_babbage_with_prob.csv')
df = pd.read_csv('calibrate/calibrated_b_mean_results.csv')

# group_by_all_df = df.groupby(['emotion', 'emotion_word', 'race', 'template', 'gender']).agg(
#     avg_intensity=pd.NamedAgg(column='predicted_intensity', aggfunc=statistics.mean)
# )

group_by_all_df = df.groupby(['emotion', 'emotion_word', 'race', 'template', 'gender']).agg(
    avg_intensity=pd.NamedAgg(column='calibrated_intensity', aggfunc=statistics.mean)
)

group_by_all_df.to_csv('analysis/babbage/group_by_all_table_calibrated_b_mean.csv')



# Get ttest result for gender bias
group_by_all_df = pd.read_csv('analysis/babbage/group_by_all_table_calibrated_b_mean.csv', names=['emotion', 'emotion_word', 'race', 'template', 'gender', 'predicted_intensity'], header=0)

pivoted_group_by_all_df = group_by_all_df.pivot(columns='gender', values='predicted_intensity')
female_predictions = pivoted_group_by_all_df.female.values.tolist()
female_predictions = [x for x in female_predictions if str(x) != 'nan']
male_predictions = pivoted_group_by_all_df.male.values.tolist()
male_predictions = [x for x in male_predictions if str(x) != 'nan']

result = stats.ttest_rel(female_predictions, male_predictions)
print('number of pairs --->', len(male_predictions))
print('----->ttest result for gender bias: ', result)


# Get ttest result for race bias
group_by_all_df = pd.read_csv('analysis/babbage/group_by_all_table_calibrated_b_mean.csv', names=['emotion', 'emotion_word', 'race', 'template', 'gender', 'predicted_intensity'], header=0)

pivoted_group_by_all_df = group_by_all_df.pivot(columns='race', values='predicted_intensity')
euro_predictions = pivoted_group_by_all_df['European'].values.tolist()
euro_predictions = [x for x in euro_predictions if str(x) != 'nan']
aa_predictions = pivoted_group_by_all_df['African-American'].values.tolist()
aa_predictions = [x for x in aa_predictions if str(x) != 'nan']

result = stats.ttest_rel(euro_predictions, aa_predictions)
print('number of pairs --->', len(aa_predictions))
print('----->ttest result for race bias: ', result)


# Baseline
# ----->ttest result for gender bias:  TtestResult(statistic=-3.2696152157456146, pvalue=0.001212017487193234, df=279)
# ----->ttest result for race bias:  TtestResult(statistic=-3.17367329226954, pvalue=0.001673584987359671, df=279)

# W multiply
# ----->ttest result for gender bias:  TtestResult(statistic=4.9427118299854795, pvalue=1.3310898800370981e-06, df=279)
# ----->ttest result for race bias:  TtestResult(statistic=2.701587119184242, pvalue=0.007323465944011612, df=279)

# W mean
# ----->ttest result for gender bias:  TtestResult(statistic=3.277098487808554, pvalue=0.0011815152946658724, df=279)
# ----->ttest result for race bias:  TtestResult(statistic=5.911042467664342, pvalue=9.897652700110067e-09, df=279)

# b
# ----->ttest result for gender bias:  TtestResult(statistic=8.793491091505208, pvalue=1.5284371182453648e-16, df=279)
# ----->ttest result for race bias:  TtestResult(statistic=10.396693678785171, pvalue=1.2992210937870004e-21, df=279)

# b mean
# ----->ttest result for gender bias:  TtestResult(statistic=1.2312094647402876, pvalue=0.21928136906802934, df=279)
# ----->ttest result for race bias:  TtestResult(statistic=5.638961629567068, pvalue=4.1902783312088014e-08, df=279)
