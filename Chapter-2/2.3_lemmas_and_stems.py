import spacy

# lemmatization is reducing words to their root forms
nlp = spacy.load('en')
doc = nlp("he was running late")
for token in doc:
    print('{} -> {}'.format(token, token.lemma_))
