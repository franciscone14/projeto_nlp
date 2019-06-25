import pandas as pd
import numpy as np
import re
import os
import spacy
from nlputils import lexical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# Modelo ja baixado
# import subprocess
# command = "python3 -m spacy download pt_core_news_sm".split()
# subprocess.call(command)
nlp = spacy.load('../models/pt_core_news_sm-2.1.0')
normalizer = lexical.Preprocessing()

class Preprocessing:
    def __init__(self):
        #carrega o corpus
        self.dataframe = pd.read_csv('../data/dados.csv')
        #aplica a funcao normalize
        self.dataframe['normalize'] = self.dataframe['text'].apply(self.normalize)
        #carrega os 4 dicionarios, insere a respectiva lista de palavras ofensivas em cada um
        self.carregarDicionarios()
   
    #as quatro funcoes abaixo retiram lixos encontrados nos textos
    def remove_username(self, tweet):
        return re.sub('@[^\s]+','',tweet)

    def remove_end_of_line(self, tweet):
        return tweet.replace('\n', '').replace('RT', '')

    def remove_duplicate_letters(self, tweet):
        # ([^rs])  - Qualquer letra que não r ou s
        # (?=\1+)  - Que se repita uma vez ou mais
        # |(rr)    - Ou dois r's
        # (?=r+)   - Que tenham mais r's à frente
        # |(ss)    - Ou dois s's
        # (?=s+)   - Que tenham mais s's à frente
        regex = r"([^rs])(?=\1+)|(rr)(?=r+)|(ss)(?=s+)"
        tweet = re.sub(regex, '', tweet, 0)
        return tweet
    
    def normalize(self, tweet):
        tweet = self.remove_username(tweet)
        tweet = self.remove_end_of_line(tweet)
        tweet = self.remove_duplicate_letters(tweet)
        tweet = normalizer.lowercase(tweet)
        tweet = normalizer.remove_punctuation(tweet)
        tokens = normalizer.tokenize_words(tweet)
        #tokens = [token for token in tokens if token not in english_stopwords]
        return ' '.join(tokens)
    
    #define para cada texto sua polaridade, sendo 1 para ofensivo e 0 para nao ofensivo
    def tag(self, tweet):
        #verifica se texto contem alguma palavra de hate_certeira
        #se sim, retorna 1
        for hate in self.hate_certeiras:
            if hate in tweet:
                return 1

        doc = nlp(tweet)
        #verifica se no texto há um verbo ofensivo e (um adjetivo ofensivo ou uma palavra ofensiva)
        #se sim, retorna 1, se não retorna 0
        for token in doc:
            if token.pos_ in ['VERB', 'AUX', 'DET']:
                if token.lemma_ in self.hate_verbo:
                    for hate in self.hate_adj:
                        if hate in tweet:
                            return 1
                    for hate in self.hate_words:
                        if hate in tweet:
                            return 1
                elif token.lemma_ in self.hate_adj:
                    return 1
            elif token.lemma_ in self.hate_adj:
                return 1
        return 0

    def split(self, probability=0.8):
        #divide os conjuntos de treinamento e de teste
        if probability < np.random.rand():
            return 'train'
        return 'test'
    def sett(self):
        my_set = []
        for i in range(0, len(self.dataframe['text'])):
            my_set.append(self.split())
        self.dataframe['set'] = my_set
        self.dataframe['tag'] = self.dataframe['normalize'].apply(self.tag)
    
    def feature_extraction(self):
        """
            Extracts the main features in the text, which are the review itself and the text polarity
            Keywords Arguments:
            * dataframe: pd.DataFrame - A Pandas DataFrame object
            Returns:
            Tuple in form (X, X_test, train_classes, teste_classes), where X and X_test are tranformers objects
        """
        train_reviews = self.dataframe[self.dataframe['set'] == 'train']['normalize'].values.tolist()
        train_classes = self.dataframe[self.dataframe['set'] == 'train']['tag'].values.tolist()
        test_reviews = self.dataframe[self.dataframe['set'] == 'test']['normalize'].values.tolist()
        test_classes = self.dataframe[self.dataframe['set'] == 'test']['tag'].values.tolist()
        
        self.transformer = TfidfVectorizer()
        self.transformer.fit(train_reviews)
        X = self.transformer.transform(train_reviews)
        X_test = self.transformer.transform(test_reviews)

        return (X, X_test, train_classes, test_classes)
    #realiza o treinamento do modelo
    def train(self):
        self.X, self.X_test, self.train_classes, self.test_classes = self.feature_extraction()
        self.classifier_lr = LogisticRegression(n_jobs=3)
        print("Trainning...")
        self.classifier_lr.fit(self.X, self.train_classes)
        print("Finished !")
        
#retorna a acuracia em porcentagem
    def accuracy(self):
        return (accuracy_score(self.test_classes, self.classifier.predict(self.X_test)) * 100)
    
    #carrega os 4 dicionarios com suas respectivas palavras ofensivas
    def carregarDicionarios(self):
        self.hate_certeiras = [
                    'feminazi','gayzista', 'viadismo', 'homossexualismo', 'viadagem', 'gayzismo', 'favelado',
                    'gayzinho', 'sapatona', 'sapatão', 'bixa', 'verme', 'negrice', 'negrisse', 'negraiada','abortistas',
                    'rameira', 'crioula', 'crioulas', 'crioulos', 'crioulo','tições', 'sangues ruins', 'Bambis', 'boiola',
                    'bichona', 'biba', 'traveco', 'travesti'
                ]

        self.hate_verbo = [
                    'matar', 'exterminar', 'odiar', 'queimar', 'atirar', 'esfaquear', 'bater', 'apanhar', 'roubar',
                    'estuprar', 'ser', 'comer',
                ]

        self.hate_adj = [
                    'safado','imbecil', 'imbecis', 'cachaceiro', 'babaca', 'idiota', 'puta', 'puto', 'arrombado','arrombada',
                    'canalha', 'vagabundo', 'vagabunda','burra', 'burro', 'maldito', 'maldita', 'neguim', 'neguinho', 
                    'favelados', 'faveladas', 'favelada', 'maconheiro', 'maconhero', 'viado'
                ]

        self.hate_words = [
                    'putaria','bosta','cuzao', 'fdp', 'filho da puta',
                    'merda', 'escroto', 'transfobia', 'gordo', 'canalha',
                    'ditadura','gorda'
                ]
   
    def verificar(self, text):
        self.transformer.fit(text)
        instance = self.transformer.transform([self.normalize(text)])
        return self.classifier_lr.predict(instance)
    