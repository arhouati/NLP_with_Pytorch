import spacy
from nltk.tokenize import TweetTokenizer

nlp = spacy.load('en')
text = "Mary, don't slap the green witch"
print([str(token) for token in nlp(text.lower())])

tweet = "Snow white and the seven Degrees" \
        "#MakeAMovieCold @midnigth:)"

tokenizer = TweetTokenizer()
print(tokenizer.tokenize(tweet))