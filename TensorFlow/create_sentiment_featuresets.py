
# coding: utf-8

# In[ ]:

import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle # saves objects to disk
from collections import Counter # dict subclass for counting hashable objects
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
hm_lines = 100000 # read first hm_lines number of lines per file


# In[2]:

# this command will load python code into this cell
# %load filename.py


# In[3]:

def create_lexicon(pos,neg):

    lexicon = []
    with open(pos,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l) # supposing pos and neg corpora are already all in lowercase
            lexicon += list(all_words)

    with open(neg,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon] # lemmatize --> synonyms will be replaced with same word
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        #print(w_counts[w])
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print(len(l2))
    return l2


# In[4]:

def sample_handling(sample,lexicon,classification):

    featureset = []
    # [ [ [0,0,1,0,1], 0] , [ [0,1,0,0,0], 1]...]

    with open(sample,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower()) # tokenizes sentence into component words
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            featureset.append([features,classification])

    return featureset


# In[5]:

def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling('pos.txt',lexicon,[1,0])
    features += sample_handling('neg.txt',lexicon,[0,1])
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x,train_y,test_x,test_y


# In[ ]:

if __name__ == '__main__':
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels('/path/to/pos.txt','/path/to/neg.txt')
    # if you want to pickle this data:
    with open('/path/to/sentiment_set.pickle','wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)

