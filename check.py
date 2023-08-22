import pandas as pd 

#check that the film (based on columnName) in a csv file is also in other csv file
def checkAllIdMatch(df1,df2,columnName):
    print(f'df1 length:{len(df1[columnName])}', f'df2 length:{len(df2[columnName])}',end=" ")
    count = 0
    for i in df1[columnName]:
        found = False
        for j in df2[columnName]:
            if i == j:
                found = True
                break
        
        if found == False:
            count += 1
            print(f'{i} in not in df2')

    if count == 0:
        print('All match')

        
#remove duplicate film
def getUniqueMovies(df1,filename):
    temp1 = set()
    ls = []
    for i in range(len(df1["id"])):
        if df1['id'][i] in temp1:
            pass
        else:
            ls.append(df1.iloc[i])
            temp1.add(df1['id'][i])

    df = pd.DataFrame(ls)
    df = df[df.columns[1:]]
    df.to_csv(f"{filename}.csv")

#getUniqueMovies(pd.read_csv('./new_credits.csv'),'new_credits')
#getUniqueMovies(pd.read_csv('./new_metadata.csv'),'new_metadata')
#getUniqueMovies(pd.read_csv('./new_keywords.csv'),'new_keywords')

checkAllIdMatch(pd.read_csv('./data/new_metadata.csv'),pd.read_csv('./data/new_credits.csv'),'id')
checkAllIdMatch(pd.read_csv('./data/new_metadata.csv'),pd.read_csv('./data/new_keywords.csv'),'id')
checkAllIdMatch(pd.read_csv('./data/new_metadata.csv'),pd.read_csv('./data/new_subtitles.csv'),'imdb_id')
