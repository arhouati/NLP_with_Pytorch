import spacy

# part of speech tagging
nlp = spacy.load('en')
doc = nlp("Mary slapped the green witch.")
for token in doc:
    print('{} -> {}'.format(token, token.pos_))