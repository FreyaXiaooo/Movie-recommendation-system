import pandas as pd 

#filter file1 based on the column of file2
def filter(file1,file2,column_name):
    df1=pd.read_csv(f'{file1}.csv',usecols=[column_name])
    df2=pd.read_csv(f'{file2}.csv')
    ls = []

    temp = 0

    for i in range(len(df2[column_name])):
        temp += 1
        print(f'{round(temp*100/len(df2[column_name]),4)}%')
        for j in range(len(df1[column_name])):
            if df2[column_name][i] == df1[column_name][j]:
                ls.append(df2.iloc[i])


    df = pd.DataFrame(ls)
    df.to_csv(f"new_{file1}.csv")


#from keywords.cvs remove all movies that has an id which is not in new_metadata.cvs
#filter("keywords","new_metadata",'id')

temp = set(pd.read_csv('new_metadata.csv',usecols=['imdb_id'])['imdb_id'])
temp2 = set(pd.read_csv('data/movies_subtitles.csv',usecols=['imdb_id'])['imdb_id'])

print(temp2)