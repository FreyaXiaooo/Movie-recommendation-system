import pandas as pd 

"""
df = pd.read_csv('movies_subtitles.csv')

ls = []
for i in range(int(len(df['imdb_id']))):
    #remove invalid rows
    if pd.isna(df['imdb_id'][i]) == False and pd.isna(df['text'][i]) == False:
        ls.append(df.iloc[i])

df = pd.DataFrame(ls)
#remove first 2 columns
df = df[df.columns[2:]]
#group rows that has the same imdb_id
result_df = df.groupby('imdb_id')['text'].agg('\n'.join).reset_index()
#keep in mind the new df has an extra column (the first one), so when we load it we need to skip the first one

result_df.to_csv(f"new_subtitles.csv")
"""


#read data
subtitle_df = pd.read_csv('data/new_subtitles.csv',usecols=['imdb_id','text'])
credits_df = pd.read_csv('data/new_credits.csv',usecols=['cast','crew','id'])
keywords_df = pd.read_csv('data/new_keywords.csv',usecols=['keywords','id'])
metadata_df = pd.read_csv('data/new_metadata.csv',usecols=['title','id','imdb_id'])

#merge 4 dataset into one
merged_df = pd.merge(metadata_df, credits_df, on='id', how='inner')
merged_df = pd.merge(merged_df, keywords_df, on='id', how='inner')
merged_df = pd.merge(merged_df, subtitle_df, on='imdb_id', how='inner')
merged_df.to_csv(f"data/final_dataset.csv")
