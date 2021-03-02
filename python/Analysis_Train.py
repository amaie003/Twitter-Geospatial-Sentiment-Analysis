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

# def read_file(path):
#     text = []
#     with open(path, encoding="utf-8") as f:
#         f_csv = csv.reader(f)
#         headers = next(f_csv)
#         for row in f_csv:
#             text.append(row[0])
#     return text


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
            lemmatized_sentence.append(wordnet.lemmatize(word.lower(),pos='a'))
        elif tag.startswith('R'):
            lemmatized_sentence.append(wordnet.lemmatize(word.lower(),pos='r'))
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

def deal_trainset(stop_words, english_punctuations):
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for pos_sen in positive_tweets:
        positive_cleaned_tokens_list.append(Cleaner(Tokenization(pos_sen), stop_words, english_punctuations))
    for neg_sen in negative_tweets:
        negative_cleaned_tokens_list.append(Cleaner(Tokenization(neg_sen), stop_words, english_punctuations))

    print(positive_cleaned_tokens_list)

    pos_model = []
    for pos_sen in positive_cleaned_tokens_list:
        pos_model.append(dict([word, True] for word in pos_sen))
    # print(pos_model)

    pos_dataset = [(pos_dict,"Positive") for pos_dict in pos_model]

    # print(pos_dataset)

    neg_model = []
    for neg_sen in negative_cleaned_tokens_list:
        neg_model.append(dict([word, True] for word in neg_sen))
    neg_dataset = [(neg_dict,"Negative") for neg_dict in neg_model]
    return pos_dataset, neg_dataset

def train(pos_dataset,neg_dataset):
    train_set = pos_dataset + neg_dataset
    classifier = NaiveBayesClassifier.train(train_set)
    with open('../model/NaiveBayesModel.pickle', 'wb') as f:
        pickle.dump(classifier,f)
    # return classifier

if __name__ == '__main__':
    # args = sys.argv
    # print(args)

    # input_path = args[1]
    # text = read_file(input_path)

    #stop_words, punctuation
    stop_words = stopwords.words('english')
    english_punctuations = []
    punc = string.punctuation
    for pun in punc:
        english_punctuations.append(pun)

    pos_dataset,neg_dataset = deal_trainset(stop_words, english_punctuations)

    train(pos_dataset, neg_dataset)