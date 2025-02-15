import pandas as pd

path1 = "test.csv"
df1 = pd.read_csv(path1)

path2 = "full_test.csv"
df2 = pd.read_csv(path2)

merged_df = pd.merge(df1, df2, on='id', how='inner')
merged_df = merged_df.rename(columns={'url_x': 'url'}).drop(columns=['url_y'])
merged_df = merged_df.rename(columns={'title_x': 'title'}).drop(columns=['title_y'])
merged_df = merged_df.drop(columns=['classify_x', 'classify_y'])
merged_df.to_csv('output.csv', index=False)
