import pandas as pd
import statistics

# babbage
# df = pd.read_csv('predicted_results_eec_babbage_with_prob.csv')

# emotion_word_avg = df.groupby(['emotion', 'emotion_word']).agg(
#     avg_intensity=pd.NamedAgg(column='predicted_intensity', aggfunc=statistics.mean)
# )
# emotion_word_avg.to_csv('analysis/babbage/emotion_word_avg_table.csv')

# emotion_word_avg_by_gender = df.groupby(['emotion', 'gender']).agg(
#     avg_intensity=pd.NamedAgg(column='predicted_intensity', aggfunc=statistics.mean)
# )
# emotion_word_avg_by_gender.to_csv('analysis/babbage/emotion_word_avg_gender_table.csv')

# emotion_word_avg_by_race = df.groupby(['emotion', 'race']).agg(
#     avg_intensity=pd.NamedAgg(column='predicted_intensity', aggfunc=statistics.mean)
# )
# emotion_word_avg_by_race.to_csv('analysis/babbage/emotion_word_avg_race_table.csv')

# _OLD
# babbage
# df = pd.read_csv('predicted_results_eec_babbage.csv')

# emotion_word_avg = df.groupby(['emotion', 'emotion_word']).agg(
#     avg_intensity=pd.NamedAgg(column='predicted_intensity', aggfunc=statistics.mean)
# )
# emotion_word_avg.to_csv('analysis/babbage/emotion_word_avg_table.csv')

# emotion_word_avg_by_gender = df.groupby(['emotion', 'gender']).agg(
#     avg_intensity=pd.NamedAgg(column='predicted_intensity', aggfunc=statistics.mean)
# )
# emotion_word_avg_by_gender.to_csv('analysis/babbage/emotion_word_avg_gender_table.csv')

# emotion_word_avg_by_race = df.groupby(['emotion', 'race']).agg(
#     avg_intensity=pd.NamedAgg(column='predicted_intensity', aggfunc=statistics.mean)
# )
# emotion_word_avg_by_race.to_csv('analysis/babbage/emotion_word_avg_race_table.csv')


# davinci
# df = pd.read_csv('predicted_results_eec_davinci.csv')

# emotion_word_avg = df.groupby(['emotion', 'emotion_word']).agg(
#     avg_intensity=pd.NamedAgg(column='predicted_intensity', aggfunc=statistics.mean)
# )
# emotion_word_avg.to_csv('analysis/davinci/emotion_word_avg_table.csv')

# emotion_word_avg_by_gender = df.groupby(['emotion', 'gender']).agg(
#     avg_intensity=pd.NamedAgg(column='predicted_intensity', aggfunc=statistics.mean)
# )
# emotion_word_avg_by_gender.to_csv('analysis/davinci/emotion_word_avg_gender_table.csv')

# emotion_word_avg_by_race = df.groupby(['emotion', 'race']).agg(
#     avg_intensity=pd.NamedAgg(column='predicted_intensity', aggfunc=statistics.mean)
# )
# emotion_word_avg_by_race.to_csv('analysis/davinci/emotion_word_avg_race_table.csv')
