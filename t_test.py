import pandas as pd
import scipy.stats as stats
import statistics

# Get average predicted intensity
df = pd.read_csv('predicted_results_eec_babbage.csv', names=['template', 'emotion', 'emotion_word', 'gender', 'race', 'predicted_intensity'], header=0)

group_by_all_df = df.groupby(['emotion', 'emotion_word', 'race', 'template', 'gender']).agg(
    avg_intensity=pd.NamedAgg(column='predicted_intensity', aggfunc=statistics.mean)
)

group_by_all_df.to_csv('analysis/babbage/group_by_all_table.csv')



# Get ttest result for gender bias
group_by_all_df = pd.read_csv('analysis/babbage/group_by_all_table.csv', names=['emotion', 'emotion_word', 'race', 'template', 'gender', 'predicted_intensity'], header=0)

pivoted_group_by_all_df = group_by_all_df.pivot(columns='gender', values='predicted_intensity')
female_predictions = pivoted_group_by_all_df.female.values.tolist()
female_predictions = [x for x in female_predictions if str(x) != 'nan']
male_predictions = pivoted_group_by_all_df.male.values.tolist()
male_predictions = [x for x in male_predictions if str(x) != 'nan']

result = stats.ttest_ind(female_predictions, male_predictions)
print('number of pairs --->', len(male_predictions))
print('----->ttest result for gender bias: ', result)


# Get ttest result for race bias
group_by_all_df = pd.read_csv('analysis/babbage/group_by_all_table.csv', names=['emotion', 'emotion_word', 'race', 'template', 'gender', 'predicted_intensity'], header=0)

pivoted_group_by_all_df = group_by_all_df.pivot(columns='race', values='predicted_intensity')
euro_predictions = pivoted_group_by_all_df['European'].values.tolist()
euro_predictions = [x for x in euro_predictions if str(x) != 'nan']
aa_predictions = pivoted_group_by_all_df['African-American'].values.tolist()
aa_predictions = [x for x in aa_predictions if str(x) != 'nan']

result = stats.ttest_ind(euro_predictions, aa_predictions)
print('number of pairs --->', len(aa_predictionss))
print('----->ttest result for race bias: ', result)


# ----->ttest result for gender bias:  Ttest_indResult(statistic=-1.0139005407428943, pvalue=0.31106963896883105)
# ----->ttest result for race bias:  Ttest_indResult(statistic=-1.8489069857292841, pvalue=0.06499999844745175)