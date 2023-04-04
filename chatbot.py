#Code edited from Neural Nine Youtube video Intelligent AI Chatbot in Python
#https://www.youtube.com/watch?v=1lwddP0KUEg 

import random 
import json 
import pickle
import numpy as np 
import nltk 
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

#tokenizes the input sentence and returns a lemmatized list of the words 
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#returns a numpy array representation of the input sentence 
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#predicts the intent of the sentence 
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERR_THRESHOLD = 0.25
    result = [[i,r] for i,r in enumerate(res) if r > ERR_THRESHOLD]

    result.sort(key = lambda x:x[1], reverse = True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

#if an intent of the user input is found this is returned, otherwise the user 
#is asked to repeat the question
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
        else:
            result = "I'm sorry I don't understand what you mean. Could you please rephrase the question?"
    return result

print("Go! Bot is running")

# while True:
#     message = input("")
#     ints = predict_class(message)
#     res = get_response(ints, intents)
#     print(res)
