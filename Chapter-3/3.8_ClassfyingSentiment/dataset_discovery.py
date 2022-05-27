from torch.utils.data import DataLoader

from dataset.review_dataset import ReviewDataset

if __name__ == "__main__":

    dataset_review = ReviewDataset.load_dataset_and_make_vectorize(review_csv='./data/raw_split.csv')

    print("Train Dataset size :", dataset_review.train_size)
    print("Train Dataset examples ::")
    print(dataset_review.train_df.head())

    print("#"*20)

    print("Valid Dataset size :", dataset_review.val_size)
    print("Valid Dataset examples ::")
    print(dataset_review.val_df.head())

    print("#" * 20)

    print("Evaluation Dataset size :", dataset_review.test_size)
    print("Evaluation Dataset examples ::")
    print(dataset_review.test_df.head())






