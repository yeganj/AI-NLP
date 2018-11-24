#module for building LSTM 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
from keras.models import model_from_json

# set seeds for reproducability
from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os 
import random

from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 

def get_sequence_of_tokens(corpus, tokenizer):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
        
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

def generate_padded_sequences(input_sequences, total_words):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
        
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=input_len))
        
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
        
    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
        
    return model

def get_distinct_words(corpus):
    all_words = []
    for i in range(0, len(corpus)):
        all_words.append(corpus[i].split())
    flat_list = [item for sublist in all_words for item in sublist]

    s = {*flat_list}
    return [*s]

def generate_embeddings(corpus, dictionary):
    # first we find the longest line
    max_len = 0
    for line in corpus:
        if len(line) > max_len:
            max_len = len(line)
        #cap the max at 20
        #if max_len > 19:
        #    max_len = 20
        #    break

    embeddings = []
    for i in range(0, len(corpus)):
        words = corpus[i].split()
        words_index = [-1] * max_len
        for j in range(0, len(words)):
            words_index[j] = dictionary.index(words[j])
        embeddings.append(words_index)
    return np.array(embeddings), max_len

def create_data(embeddings):
    i = len(embeddings)
    ones = np.ones((i, 1))
    complete = np.hstack((embeddings,ones))

    incomplete = []
    for i in range(0, len(embeddings)):
        end = list(embeddings[i]).index(-1) - 2
        if end > 1:
            index = random.randint(1, end)
            #new_embed = list(embeddings[i])[:index] + [-1]*(len(embeddings)-index)
            new_embed = list(embeddings[i][:index]) + [-1]*(len(list(embeddings[i]))-index)
            incomplete.append(new_embed)

    for i in range(0, len(embeddings)):
        end = list(embeddings[i]).index(-1) - 2
        if end > 1:
            index = random.randint(1, end)
            #new_embed = list(embeddings[i])[:index] + [-1]*(len(embeddings)-index)
            new_embed = list(embeddings[i][:index]) + [-1]*(len(list(embeddings[i]))-index)
            incomplete.append(new_embed)

    incomplete = np.array(incomplete)

    i = len(incomplete)
    zeros = np.zeros((i, 1))
    incomplete = np.hstack((incomplete,zeros))
    all_data = np.concatenate((complete, incomplete))

    np.random.shuffle(all_data)
    arr = np.array_split(all_data, 2)
    return arr[0], arr[1]

def get_x_y(data):
    return data[:,:-1], data[:,-1]

def generate_text(seed_text, next_words, model, max_sequence_len, tokenizer, clf, dictionary, max_len):

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()

def generate_text_predictor(seed_text, next_words, model, max_sequence_len, tokenizer, clf, dictionary, max_len):

    predicted_seed_text = ''
    is_complete = False
    i = 0
    while is_complete == False and i < max_len:
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
        is_complete = test_sequence(seed_text, dictionary, clf, max_len)
        i += 1
        #print(seed_text)
    return seed_text.title()

def test_sequence(seed, dictionary, clf, max_len):

    seed_text = seed.split()
    seed_array = [-1] * max_len
    
    for i in range(0, len(seed_text)):
        seed_array[i] = dictionary.index(seed_text[i])

    result = clf.predict([seed_array])
    
    if int(result[0]) == 1:
        return True
    return False


def main():

    tokenizer = Tokenizer()
    curr_dir = '../nyt-comments/'
    all_headlines = []
    for filename in os.listdir(curr_dir):
        if 'ArticlesJan' in filename or 'ArticlesFeb' in filename:
            print(filename)
            article_df = pd.read_csv(curr_dir + filename)
            all_headlines.extend(list(article_df.headline.values))


    #print(all_headlines[:10])
    print(len(all_headlines))

    print("creating corpus")
    corpus = [clean_text(x) for x in all_headlines]
    #print(len(corpus))

    print("creating dictionary")
    dictionary = get_distinct_words(corpus)
    #print(dictionary[:10])

    print("creating embeddings")
    embeddings, max_len = generate_embeddings(corpus, dictionary)

    #print(embeddings[:10])
   
    print("creating train, test data")
    train, test = create_data(embeddings)

    X_train, y_train = get_x_y(train)
    X_test, y_test = get_x_y(test)

    print("creating classifier")
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
    print("Classifier score on training data: {}".format(clf.score(X_train, y_train)))
    print("Classifier score on test data: {}".format(clf.score(X_test, y_test)))


    print("get sequence of tokens")
    inp_sequences, total_words = get_sequence_of_tokens(corpus, tokenizer)

    print("generate padded sequences")
    predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences, total_words)

    #print("creating model")
    #model = create_model(max_sequence_len, total_words)
    #print(model.summary())

    #model.fit(predictors, label, epochs=100, verbose=2)

    # serialize model to JSON
    #model_json = model.to_json()
    #with open("model.json", "w") as json_file:
    #    json_file.write(model_json)
    #    # serialize weights to HDF5
    #    model.save_weights("model.h5")
    #print("Saved model to disk")

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
 
    print("Loaded model from disk")


    print("\n\nResults:")

    print(generate_text_predictor("united states", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))
    print(generate_text("united states", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))

    print(generate_text_predictor("india and china", 4, model, max_sequence_len, tokenizer, clf, dictionary, max_len))
    print(generate_text("india and china", 4, model, max_sequence_len, tokenizer, clf, dictionary, max_len))

    print(generate_text_predictor("new york", 4, model, max_sequence_len, tokenizer, clf, dictionary, max_len))
    print(generate_text("new york", 4, model, max_sequence_len, tokenizer, clf, dictionary, max_len))
    
    print(generate_text_predictor("donald trump", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))
    print(generate_text("donald trump", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))


    print(generate_text_predictor("science and technology", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))
    print(generate_text("science and technology", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))

    print(generate_text_predictor("immigration", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))
    print(generate_text("immigration", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))

    print(generate_text_predictor("north korea", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))
    print(generate_text("north korea", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))


    print(generate_text_predictor("japan", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))
    print(generate_text("japan", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))

    print(generate_text_predictor("barack obama", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))
    print(generate_text("barack obama", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))


    print(generate_text_predictor("hillary clinton", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))
    print(generate_text("hillary clinton", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))

    print(generate_text_predictor("canada", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))
    print(generate_text("canada", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))

    print(generate_text_predictor("nafta", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))
    print(generate_text("nafta", 5, model, max_sequence_len, tokenizer, clf, dictionary, max_len))

if __name__=="__main__":
    main()
