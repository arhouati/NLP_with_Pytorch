import spacy


# Notice that spacy and nltk provide also convenient methods to get n-grams from text
def n_grams(tokens: list, n: int) -> list:
    """
    takes tokens of text, returns a list of n-grams

    :param tokens: list
    :param n: int
    :return: list
    """
    return [list(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


nlp = spacy.load('en')
text = "Mary, don't slap the green witch"
cleaned = nlp( text.lower() )
print(n_grams(cleaned, 3) )

