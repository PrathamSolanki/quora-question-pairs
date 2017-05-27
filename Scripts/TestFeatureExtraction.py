import numpy as np
import pandas as pd


# Loading Train Data
test_df = pd.read_csv('../../Data/test.csv')

# Aim is to Extract features that can differentiate whether the pair of questions are duplicate or not

from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
import string
from nltk.tag import pos_tag

eng_stopwords = set(stopwords.words('english'))
punctuations = list(string.punctuation)


# Word tokenizer

def get_words(question):
    #question = question.decode('utf-8').replace(u'\u014c\u0106\u014d','-')
    return word_tokenize(question.lower())

test_df["words_ques1"] = test_df['question1'].apply(lambda x: get_words(str(x)))
test_df["words_ques2"] = test_df['question2'].apply(lambda x: get_words(str(x)))

# Unigrams (removing stop words and punctuations, selecting only unique words)

def get_unigrams(words):
    unigrams = [] 
    for word in words:
        if word not in unigrams and word not in eng_stopwords and word not in punctuations:
            unigrams.append(word)
            
    return unigrams

test_df["unigrams_ques1"] = test_df['words_ques1'].apply(lambda x: get_unigrams(x))
test_df["unigrams_ques2"] = test_df['words_ques2'].apply(lambda x: get_unigrams(x))

# Stemming the unigrams (Porter Stemmer Bugged in 3.2.2)

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def stem_words(words):
    return [stemmer.stem(word) for word in words]

test_df["unigrams_ques1"] = test_df['unigrams_ques1'].apply(lambda x: stem_words(x))
test_df["unigrams_ques2"] = test_df['unigrams_ques2'].apply(lambda x: stem_words(x))

# Common unigrams between the question pairs

unigrams_common = []

def get_common_unigrams(row):
    unigrams_common.append(set(row["unigrams_ques1"]).intersection(set(row["unigrams_ques2"])))
    return len( set(row["unigrams_ques1"]).intersection(set(row["unigrams_ques2"])) )

test_df["unigrams_common_count"] = test_df.apply(lambda row: get_common_unigrams(row),axis=1)
test_df["unigrams_common"] = unigrams_common

# Ratio of common unigrams to the total number of words between question pairs

def get_common_unigram_ratio(row):
    return float(row["unigrams_common_count"]) / max(len( set(row["unigrams_ques1"]).union(set(row["unigrams_ques2"])) ),1)

test_df["unigrams_common_ratio"] = test_df.apply(lambda row: get_common_unigram_ratio(row), axis=1)

# Simliarly for bigrams

def get_bigrams(question):
    return [i for i in ngrams(question, 2)]

def get_common_bigrams(row):
    return len( set(row["bigrams_ques1"]).intersection(set(row["bigrams_ques2"])) )

def get_common_bigram_ratio(row):
    return float(row["bigrams_common_count"]) / max(len( set(row["bigrams_ques1"]).union(set(row["bigrams_ques2"])) ),1)

test_df["bigrams_ques1"] = test_df["unigrams_ques1"].apply(lambda x: get_bigrams(x))
test_df["bigrams_ques2"] = test_df["unigrams_ques2"].apply(lambda x: get_bigrams(x)) 
test_df["bigrams_common_count"] = test_df.apply(lambda row: get_common_bigrams(row),axis=1)
test_df["bigrams_common_ratio"] = test_df.apply(lambda row: get_common_bigram_ratio(row), axis=1)

# And similiarly for trigrams

def get_trigrams(question):
    return [i for i in ngrams(question, 3)]

def get_common_trigrams(row):
    return len( set(row["trigrams_ques1"]).intersection(set(row["trigrams_ques2"])) )

def get_common_trigram_ratio(row):
    return float(row["trigrams_common_count"]) / max(len( set(row["trigrams_ques1"]).union(set(row["trigrams_ques2"])) ),1)

test_df["trigrams_ques1"] = test_df["unigrams_ques1"].apply(lambda x: get_trigrams(x))
test_df["trigrams_ques2"] = test_df["unigrams_ques2"].apply(lambda x: get_trigrams(x)) 
test_df["trigrams_common_count"] = test_df.apply(lambda row: get_common_trigrams(row),axis=1)
test_df["trigrams_common_ratio"] = test_df.apply(lambda row: get_common_trigram_ratio(row), axis=1)

# Nouns

def extract_nouns(words):
    tagged_sent = pos_tag(words)
    return [word for word,pos in tagged_sent if pos == 'NN' or 
                                                pos == 'NNS' or
                                                pos == 'NNP' or
                                                pos == 'NNPS']

test_df["nouns_ques1"] = test_df['unigrams_ques1'].apply(lambda x: extract_nouns(x))
test_df["nouns_ques2"] = test_df['unigrams_ques2'].apply(lambda x: extract_nouns(x))


def get_common_nouns(row):
    return len( set(row["nouns_ques1"]).intersection(set(row["nouns_ques2"])) )

test_df["nouns_common_count"] = test_df.apply(lambda row: get_common_nouns(row),axis=1)


def get_common_nouns_ratio(row):
    return float(row["nouns_common_count"]) / max(len( set(row["nouns_ques1"]).union(set(row["nouns_ques2"])) ),1)

test_df["nouns_common_ratio"] = test_df.apply(lambda row: get_common_nouns_ratio(row), axis=1)

# Word Match share

from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

test_df['word_match'] = test_df.apply(word_match_share, axis=1, raw=True)

# tf-idf word match share

from collections import Counter

train_qs = pd.Series(test_df['question1'].tolist() + test_df['question2'].tolist()).astype(str)

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

test_df['tfidf_test_word_match'] = test_df.apply(tfidf_word_match_share, axis=1, raw=True)


test_df.to_pickle('test_dataframe')
