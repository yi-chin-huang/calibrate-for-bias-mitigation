import pandas as pd
import scipy.stats as stats
import statistics

# Get average predicted intensity
# df = pd.read_csv('predicted_results_eec_babbage_with_prob.csv')
df = pd.read_csv('calibrate/calibrated_mean_results.csv')

# group_by_all_df = df.groupby(['emotion', 'emotion_word', 'race', 'template', 'gender']).agg(
#     avg_intensity=pd.NamedAgg(column='predicted_intensity', aggfunc=statistics.mean)
# )

group_by_all_df = df.groupby(['emotion', 'emotion_word', 'race', 'template', 'gender']).agg(
    avg_intensity=pd.NamedAgg(column='calibrated_intensity', aggfunc=statistics.mean)
)

group_by_all_df.to_csv('analysis/babbage/group_by_all_table_calibrated_mean.csv')



# Get ttest result for gender bias
group_by_all_df = pd.read_csv('analysis/babbage/group_by_all_table_calibrated_mean.csv', names=['emotion', 'emotion_word', 'race', 'template', 'gender', 'predicted_intensity'], header=0)

pivoted_group_by_all_df = group_by_all_df.pivot(columns='gender', values='predicted_intensity')
female_predictions = pivoted_group_by_all_df.female.values.tolist()
female_predictions = [x for x in female_predictions if str(x) != 'nan']
male_predictions = pivoted_group_by_all_df.male.values.tolist()
male_predictions = [x for x in male_predictions if str(x) != 'nan']

result = stats.ttest_rel(female_predictions, male_predictions)
print('number of pairs --->', len(male_predictions))
print('----->ttest result for gender bias: ', result)


# Get ttest result for race bias
group_by_all_df = pd.read_csv('analysis/babbage/group_by_all_table_calibrated_mean.csv', names=['emotion', 'emotion_word', 'race', 'template', 'gender', 'predicted_intensity'], header=0)

pivoted_group_by_all_df = group_by_all_df.pivot(columns='race', values='predicted_intensity')
euro_predictions = pivoted_group_by_all_df['European'].values.tolist()
euro_predictions = [x for x in euro_predictions if str(x) != 'nan']
aa_predictions = pivoted_group_by_all_df['African-American'].values.tolist()
aa_predictions = [x for x in aa_predictions if str(x) != 'nan']

result = stats.ttest_rel(euro_predictions, aa_predictions)
print('number of pairs --->', len(aa_predictions))
print('----->ttest result for race bias: ', result)


#baseline
# ----->ttest result for gender bias:  TtestResult(statistic=-3.2696152157456146, pvalue=0.001212017487193234, df=279)
# ----->ttest result for race bias:  TtestResult(statistic=-3.17367329226954, pvalue=0.001673584987359671, df=279)

# multiply
# ----->ttest result for gender bias:  TtestResult(statistic=4.9427118299854795, pvalue=1.3310898800370981e-06, df=279)
# ----->ttest result for race bias:  TtestResult(statistic=2.701587119184242, pvalue=0.007323465944011612, df=279)

# mean
# ----->ttest result for gender bias:  TtestResult(statistic=1.542925742744917, pvalue=0.12398223865987341, df=279)
# ----->ttest result for race bias:  TtestResult(statistic=2.0503354933751017, pvalue=0.04126601597820872, df=279)
