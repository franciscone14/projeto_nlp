import nltk
import unidecode
import string
import os

from nltk.corpus import stopwords

portuguese_stop_words = stopwords.words('portuguese')

class Preprocessing:

    def __init__(self, save_path='../data/normilized/', file_name=None):
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        self.stemmer = nltk.stem.RSLPStemmer()

        self.save_path = save_path
        self.file_name = file_name

        if(not os.path.isdir(save_path)):
            os.makedirs(save_path)

    def save(self, path=None, file_name=None, data=None):
        if not os.path.isdir(path):
            os.makedirs(path)
        
        with open(os.path.join(path, file_name), 'a') as file:
            file.write('%s \n' % data)
    
    def lowercase(self, text, save=False):
        return text.lower()

    def remove_accents(self, text, save=False):
        text = unidecode.unidecode(text)
        if save:
            path = os.path.join(self.save_path, 'accents_removed')
            file_name = self.file_name + '_no_accents.txt'
            self.save(path, file_name, text)
        return text
    
    def remove_punctuation(self, text, save=False):
        text = text.translate(str.maketrans('','', string.punctuation))
        if save:
            path = os.path.join(self.save_path, 'punctuation_removed')
            file_name = self.file_name + '_no_punct.txt'
            self.save(path, file_name, text)
        return text

    def tokenize_sentences(self, text, save=False):
        text = self.sent_tokenizer.tokenize(text)
        if save:
            path = os.path.join(self.save_path, 'tokenized_sentences')
            file_name = self.file_name + '_tokenized_sentences.txt'
            self.save(path, file_name, text)
        return text

    def tokenize_words(self, text, save=False):
        tokens = nltk.tokenize.word_tokenize(text)
        if save:
            path = os.path.join(self.save_path, 'tokenized_words')
            file_name = self.file_name + '_tokenized.txt'
            self.save(path, file_name, tokens)
        return tokens
    
    def remove_stopwords(self, tokens, save=False):
        for token in tokens:
            if token in portuguese_stop_words:
                tokens.remove(token)

        return tokens
    
    def lemmatize(self, text, save=False):
        if save:
            path = os.path.join(self.save_path, 'lemmatized')
            file_name = self.file_name + '_lemmatized.txt'
            self.save(path, file_name, text)
        else:
            return text

    def stemmize(self, tokens, save=False):
        stem = [self.stemmer.stem(word) for t in tokens for word in t ]
        if save:
            path = os.path.join(self.save_path, 'stemmized')
            file_name = self.file_name + '_stemmized.txt'
            self.save(path, file_name, stem)
        return stem
    
    def normalization_pipeline(self, text, 
        remove_accents=False, remove_punctuation=False, 
        tokenize_sentences=False, tokenize_words=False, 
        lemmatize=False, stemmize=False, save=False):
        
        text = self.remove_accents(text, save) if remove_accents else text
        text = self.tokenize_sentences(text, save) if tokenize_sentences else text
        text = self.remove_punctuation(text, save) if remove_punctuation else text
        text = self.tokenize_words(text, save) if tokenize_words else text
        text = self.lemmatize(text, False) if lemmatize else text
        text = self.stemmize(text, save) if stemmize else text
        
        return text   