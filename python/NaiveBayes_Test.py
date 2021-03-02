import sys
import numpy as np
import csv
import nltk

nltk.download('stopwords')
nltk.download('twitter_samples')
nltk.download('punkt')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
import string
import re
import math
# import pickle

def read_file(path):
    # with open(path)
    # text = np.loadtxt(path, delimiter=',', encoding='utf-8', dtype="%s,%s")
    text = []
    with open(path, encoding="gbk") as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            text.append(row[0])
    return text


def Tokenization(sentence):
    token_method = TweetTokenizer()
    token_list = token_method.tokenize(sentence)
    return token_list


def Cleaner(token_list, stop_words=(), english_punctuations=()):
    # pos_tag && Lemmatisation
    wordnet = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(token_list):
        word = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", word)
        word = re.sub(r"@[a-zA-Z0-9_]+", "", word)
        if tag.startswith('NN'):
            lemmatized_sentence.append(wordnet.lemmatize(word.lower(), pos='n'))
        elif tag.startswith('VB'):
            lemmatized_sentence.append(wordnet.lemmatize(word.lower(), pos='v'))
        elif tag.startswith('JJ'):
            lemmatized_sentence.append(wordnet.lemmatize(word.lower(), pos='a'))
        elif tag.startswith('R'):
            lemmatized_sentence.append(wordnet.lemmatize(word.lower(), pos='r'))
        else:
            lemmatized_sentence.append(word.lower())

    # remove stop_words and the punctuations
    cleaned_token_list = []
    for word in lemmatized_sentence:
        if word.lower() not in stop_words and len(word) > 0 and word not in english_punctuations:
            cleaned_token_list.append(word)
    return cleaned_token_list

def read_pos_neg_words(pos_path, neg_path):
    with open(pos_path, encoding='utf-8') as f:
        temp = f.readlines()
    all_pos_words = [word.strip('\n') for word in temp]

    with open(neg_path, encoding='utf-8') as f:
        temp = f.readlines()
    all_neg_words = [word.strip('\n') for word in temp]

    return all_pos_words, all_neg_words

def predict(sentence, all_pos_words, all_neg_words, stop_words, english_punctuations):
    cleaned_sentence = Cleaner(Tokenization(sentence), stop_words, english_punctuations)
    pos_pro_ln = 0.0
    neg_pro_ln = 0.0
    for word in cleaned_sentence:
        pos_count = 1.0
        neg_count = 1.0
        pos_denom = 2.0
        neg_denom = 2.0
        if word in all_pos_words:
            pos_count += 1.0
            pos_denom += 1.0
        if word in all_neg_words:
            neg_count += 1.0
            neg_denom += 1.0
        pos_pro_ln = pos_pro_ln + math.log(pos_count / pos_denom) + math.log(1.0 / len(all_pos_words))
        neg_pro_ln = neg_pro_ln + math.log(neg_count / neg_denom) + math.log(1.0 / len(all_neg_words))
    # print(pos_pro_ln, neg_pro_ln)
    pos_pro = math.exp(pos_pro_ln)
    neg_pro = math.exp(neg_pro_ln)
    # print('pos_pro: ', pos_pro)
    # print('neg_pro: ', neg_pro)
    print('pos_pro: ',pos_pro, 'neg_pro: ', neg_pro)
    if pos_pro > neg_pro:
        # print('Positive')
        return pos_pro, 'Positive'
    else:
        # print('Negative')
        return neg_pro, 'Negative'

if __name__ == '__main__':
    args = sys.argv
    print(args)

    input_path = args[1]
    text = read_file(input_path)

    # stop_words, punctuation
    stop_words = stopwords.words('english')
    english_punctuations = []
    punc = string.punctuation
    for pun in punc:
        english_punctuations.append(pun)

    pos_path = '../model/pos_words.txt'
    neg_path = '../model/neg_words.txt'

    all_pos_words, all_neg_words = read_pos_neg_words(pos_path, neg_path)
    # print(text[0])
    # predict(text[0], all_pos_words, all_neg_words, stop_words, english_punctuations)

    whole_text = []
    with open(input_path, encoding="gbk") as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            whole_text.append(row)
    f.close()
    # print(whole_text[0][1])

    with open("../result/naivebayes_sentiment_data.csv", "w", encoding='gbk') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "location", "score", "label"])
        for i in range(len(text)):
            score, label = predict(text[i], all_pos_words, all_neg_words, stop_words, english_punctuations)
            writer.writerows([[text[i], whole_text[i][1], str(score), label]])
    f.close()



    # print(pos_dataset,neg_dataset)

    # train(all_pos_words, all_neg_words)

    #
    # conf = SparkConf().setAppName('Analysis')
    # conf.setMaster("local")
    # sc = SparkContext(conf=conf)
    #
    # read_file(sc, input_path)

    # print('end')