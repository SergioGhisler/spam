#Paso 1
import os
from random import shuffle
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#Paso 2
ham = os.listdir('/Users/sergio/Desktop/Universidad/Inteligencia artificial/spam/venv/enron1/ham')
spam = os.listdir('/Users/sergio/Desktop/Universidad/Inteligencia artificial/spam/venv/enron1/spam')

#Paso 3
all_ham=[]
for i in range(0,ham.__len__()):
    f = open('/Users/sergio/Desktop/Universidad/Inteligencia artificial/spam/venv/enron1/ham/'+ham[i], 'rb')
    all_ham.append(tuple((f.read(),"ham")))
all_spam=[]
for i in range(0,spam.__len__()):
    f = open('/Users/sergio/Desktop/Universidad/Inteligencia artificial/spam/venv/enron1/spam/' + spam[i], 'rb')
    all_spam.append(tuple((f.read(), "spam")))


all_emails=all_ham+all_spam

#Paso 4
print (len(all_emails))

#Paso 5
shuffle(all_emails)

#Paso 7 y 8
def clean(n):
    n=unicode(n, errors='replace')
    tokens=word_tokenize(n)
    sr = stopwords.words("english")
    clean_tokens = tokens[:]
    contador = 0
    for i in clean_tokens:
        clean_tokens[contador]=i.lower()
        contador= contador+1
    for i in tokens:
        i=i.lower()
        if i in sr:
            clean_tokens.remove(i)
    return clean_tokens
lemmatizer = WordNetLemmatizer()

def token(n):

    tokens = word_tokenize(n)
    clean_tokens = tokens[:]
    contador = 0
    for i in clean_tokens:
        clean_tokens[contador]=i.lower()
        contador= contador+1
    return clean_tokens

def lemmatizeToken(t):
    new=[]

    for i in t:

        new.append(lemmatizer.lemmatize(i))
    return new
#Paso 9
def stop(n):
    sw=stopwords.words("english")
    if n in sw:
        return False
    else:
        return True
def crearDiccionario(n):

    m=n

    d = {}

    for i in m:

        d[i] = stop(i)

    return d

all_emails_definitivo=[(clean(n),spam)for(n,spam)in all_emails]




all_emails_definitivo

all_features=[(crearDiccionario(n),spam)for (n,spam) in all_emails_definitivo]

print(str(len(all_features)))




print('Obtenemos la longitud de todas las features:' +
str(len(all_features)))

from nltk import NaiveBayesClassifier, classify


def train(features):
    print("He llegado")
    train_set = features[1:4000]
    test_set = features[4001:5172]

    classifier = NaiveBayesClassifier.train(train_set)

    print('Accuracy on the training set = ' +
          str(classify.accuracy(classifier, train_set)))
    print('Accuracy of the test set = ' + str(classify.accuracy(classifier, test_set)))
    classifier.show_most_informative_features(20)


train(all_features)