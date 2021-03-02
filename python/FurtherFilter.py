import pandas as pd

df = pd.read_csv('../result/analysis_sentiment_data.csv', encoding='GBK')             #read csv file

df = df[df['text'].notnull()]                             #the text can't be null

print('input a keyword you want to search: ')

df1 = (df[df["text"].str.contains(input())])                #filtering according to input

df2=df1.drop(['text'], axis=1)                              #delete content

df2.to_csv("../dataset/3.csv", encoding='UTF-8', header=None, index=None)       #write into a new csv file
