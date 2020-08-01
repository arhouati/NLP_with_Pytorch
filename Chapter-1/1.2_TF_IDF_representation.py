from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

corpus = ['Time flies flies like an arrow',
          'Fruit flies like a banana']

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()

# one hot representation of corpus' words
sns.heatmap(tfidf, annot=True, cbar=False, yticklabels=['sentence 1', 'sentence 2'],
            xticklabels=sorted(tfidf_vectorizer.vocabulary_))
plt.show()
