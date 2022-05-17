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

    # DataLoader
    def generte_batches( dataset, batch_size, shuffle = True,
                         drop_last = True, device = "cpu"):

        """
        A generator function which wraps the Pytorch Dataloader,
        It will ensure each tensor is on the write device location
        """
        dataloader = DataLoader(dataset = dataset, batch_size = batch_size,
                                shuffle = shuffle, drop_last = drop_last)

        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield  out_data_dict






