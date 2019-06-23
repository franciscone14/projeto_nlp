import spacy

class Morpho:

    def __init__(self, language="pt_core_news_sm"):
        self.nlp = spacy.load(language)

    def tag(self, string):
        
        doc = self.nlp(string)

        tokens = []
        for token in doc:
            tokens.append((token.text, token.pos_))
        
        return tokens