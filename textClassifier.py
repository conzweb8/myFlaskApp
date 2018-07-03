#!flask/bin/python
import csv
import os
import sys

import nltk
from flask import Flask, jsonify
from flask import abort
from flask import request
from nltk import word_tokenize

app = Flask(__name__)

train_file_src = "/Users/constantin/Project/Learn NLP/nltk_data/corpora/chat_logs/all_chat.csv"
# train_file_src = "all_chat.csv"

# Get file
with open(train_file_src, 'r') as f:
    reader = csv.reader(f)
    # csv_chat_log = map(tuple, reader)
    csv_chat_log = list(reader)

# Get all words from the file
all_words = []
for sentence in csv_chat_log:
    test = word_tokenize(list(sentence)[0])
    for words in test:
        all_words.append(words)


# Features build by get all words compare to chat and count it
def enhanced_chat_features(chat):
    features = {}

    for word in all_words:
        features["contain({})".format(word)] = (word in chat)

    for word in chat.split(" "):
        features["count({})".format(word)] = all_words.count(word)

    features["length"] = len(chat)
    return features


# Create train set from features chat list
chat_list_train_set = []

for sent in csv_chat_log:
    chat_list_train_set.append((enhanced_chat_features(sent[0]), sent[1]))

# Classify using NB from train set
nl = nltk.NaiveBayesClassifier.train(chat_list_train_set)
nltk.NaiveBayesClassifier.show_most_informative_features(nl, 5)


# Available Service Listen...

# Classify text based on trained data
@app.route('/classify', methods=['POST'])
def create_task():
    if not request.json or not 'text' in request.json:
        abort(400)

    text_to_classify = request.json['text']
    print(text_to_classify)
    result = nl.classify(enhanced_chat_features(text_to_classify))
    return jsonify({'result': result}), 200


# Add new data to train data sets
@app.route('/addtrain', methods=['POST'])
def add_train_label():
    if not request.json or not 'text' in request.json:
        abort(400)

    if not request.json or not 'label' in request.json:
        abort(400)

    text_to_train = request.json['text']
    label_to_train = request.json['label']

    with open(train_file_src, 'a') as file:
        file.writelines("{},{}\n".format(text_to_train,label_to_train))
        file.close()

    print("Train '{}' with this label '{}'.".format(text_to_train, label_to_train))

    return jsonify({'result': 'OK'}), 200


# Retrain newly added trained data
@app.route('/retrain', methods=['POST'])
def retrain():
    if not request.json or not 'pswd' in request.json:
        abort(400)

    pswd = request.json['pswd']

    if pswd != 'tintinkeren':
        abort(400)

    print('restarting...')
    os.execl(sys.executable, sys.executable, *sys.argv)

    return jsonify({'retrained': 'OK'}), 200


if __name__ == '__main__':
    # app.run(host="0.0.0.0")
    app.run(debug=True)
