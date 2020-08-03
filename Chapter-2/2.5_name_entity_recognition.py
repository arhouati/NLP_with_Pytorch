import spacy


nlp = spacy.load('en_core_web_sm')
doc = nlp("Mary slapped the green witch")

# chunking is identify the noun phrase (NP) and verb phrase (VB)
for chunk in doc.noun_chunks:
    print('{} -> {}'.format(chunk, chunk.label_))

#name entity recognition
for ent in doc.ents:
    print('{} -> {}'.format(ent, ent.label_))
