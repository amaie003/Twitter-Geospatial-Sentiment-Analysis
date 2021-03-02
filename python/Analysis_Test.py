import sys
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
import pickle
import math
import codecs

def read_file(path):
    text = []
    with codecs.open(path, 'r', encoding='gbk') as f:

        f_csv = csv.reader(f)
        # print(f.readline())
        headers = next(f_csv)
        # print(headers)
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
        # else:
        #     print(word)
    return cleaned_token_list

def load_model(name):
    with open(name, 'rb') as f:
        classifier = pickle.load(f)
    f.close()
    return classifier

def predict(sentence, model, stop_words, english_punctuations):
    sen_model = dict([word, True] for word in Cleaner(Tokenization(sentence), stop_words, english_punctuations))
    pro = model.prob_classify(sen_model)._prob_dict.items()
    # print(pro)
    # print(pro[0][1])
    pos_pro = math.exp(list(pro)[0][1])
    neg_pro = math.exp(list(pro)[1][1])
    # print(sentence)
    print('pos_pro: ',pos_pro, 'neg_pro: ', neg_pro)
    if pos_pro > neg_pro:
        return pos_pro, 'Positive'
    else:
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

    name = '../model/NaiveBayesModel.pickle'
    classifier = load_model(name)

    # print(text[21])
    # pro, label = predict(text[21], classifier, stop_words, english_punctuations)
    # print(pro,label)

    whole_text = []
    with open(input_path, encoding="gbk") as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            whole_text.append(row)
    f.close()
    # print(whole_text[0][1])

    with open("../result/analysis_sentiment_data.csv", "w", encoding='gbk') as f:
        writer = csv.writer(f)
        writer.writerow(["text","location","score","label"])
        for i in range(len(text)):
            score, label = predict(text[i], classifier, stop_words, english_punctuations)
            writer.writerows([[text[i], whole_text[i][1], str(score), label]])
    f.close()