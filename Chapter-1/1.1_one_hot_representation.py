from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

corpus = ['Time flies flies like an arrow',
          'Fruit flies like a banana']

one_hot_vectorizer = CountVectorizer(binary=True)
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()

# one hot representation of corpus' words
sns.heatmap(one_hot, annot=True, cbar=False, yticklabels=['sentence 1', 'sentence 2'],
            xticklabels=sorted(one_hot_vectorizer.vocabulary_))
plt.show()