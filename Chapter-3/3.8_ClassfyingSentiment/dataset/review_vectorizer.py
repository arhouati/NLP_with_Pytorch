import string
from collections import Counter

import numpy as np
from .vocabulary import Vocabulary


class ReviewVectorizer():
    """ The Vectorize which coordinates the vocabularies and puts them to use"""

    def __init__(self, review_vocab, rating_vocab):
        """

        :param review_vocab: maps words to integers
        :param rating_vocab: maps class labels tointegers
        """
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    def vectorize(self, review: str):
        """
        Create a Collapsed one-hit vector for the review

        :param review (str): the review
        :return one_hot (np.ndarray) : the collapsed one-hot encoding
        """

        one_hot = np.zeros(len(self.review_vocab), dtype=np.float)

        for token in review.split(" "):
            if token not in string.punctuation:
                one_hot[self.review_vocab.lookup_token(token)] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        """
        Instantiate the vectorize from the dataset dataframe

        :param review_df (panda.DataFrame): the review dataset
        :param cutoff:
        :return contents (dict): the serializable dictionary
        """
        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)

        # Add ratings
        for rating in sorted(set(review_df.rating)):
            rating_vocab.add_token(rating)

        # Add top words if count > provied count (cutoff)
        word_counts = Counter()
        for review in review_df.text:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)

        return cls(review_vocab, rating_vocab)

    @classmethod
    def from_serializable(cls, contents):
        """
        Intantiate a ReviewVectorize from a serializable dictionary

        :param contents (dicts): the serializable dictionary
        :return: an instance of the ReviewVectorizer class
        """
        review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
        rating_vocab = Vocabulary.from_serializable(contents['rating_vocab'])

        return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)

    def to_serializable(self):
        """
        create the serializable dictionary caching

        :return:
            contents (dict)= The serializable dictionary
        """

        return {'review_vocab': self.review_vocab.to_serializable(),
                'rating_vocab': self.rating_vocab.to_serializable()}