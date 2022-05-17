from torch.utils.data import Dataset
import pandas as pd
from .review_vectorizer import ReviewVectorizer


class ReviewDataset(Dataset):

    def __init__(self, review_df, vectorizer):
        """
        Args:
            review_df (pandas.DataFrame) : the dataset from the script dataset_split.py
            vectorizer (ReviewVectorizer) : vectorizer instanciated from dataset (will be seen in the next section
        """

        self.review_df = review_df
        self._vectorizer = vectorizer

        self.train_df = self.review_df[self.review_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.review_df[self.review_df.split == 'val']
        self.val_size = len(self.val_df)

        self.test_df = self.review_df[self.review_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.val_size),
                             'test': (self.test_df, self.test_size) }

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorize(cls, review_csv):
        """
        :param review_csv (str): location of the dataset
        :return: an instance of ReviewDataset
        """
        review_df = pd.read_csv(review_csv)
        return cls(review_df, ReviewVectorizer.from_dataframe(review_df))

    def get_vectorizer(self):
        """
        :return: vectorizer
        """
        return self._vectorizer

    def set_split(self, split="train"):
        """
        Selects the split in the dataset using a column of dataFrame
        :param split (str): one of the values 'train', 'val' or 'test'
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """

        :param index (int): the index of the dataset point
        :return: a dict of the data point's feature (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        review_vector = self._vectorizer.vectorize(row.text)

        rating_index = self._vectorizer.rating_vocab.lookup_token(row.rating)

        return {'x_data': review_vector,
                'y_label': rating_index}

    def get_num_batches(self, batch_size):
        """
        Given a batch size, retur the number of batches in the dataset

        :param batch_size (int) : this is a paramter depeond on data and model
        :return: number of batches in dataset
        """
        return len(self) // batch_size