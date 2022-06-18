from argparse import Namespace

from tqdm import tqdm

import torch.cuda
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from dataset.review_dataset import ReviewDataset
from model.review_classifier import ReviewClassifier

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

args = Namespace(
    # Data and path informations
    frequency_cutoff = 25,
    model_state_file = "model.pth",
    review_csv = "data/raw_split.csv",
    save_dir = "model_storage/ch3/yelp",
    vectorize_file = "vectorizer.json",
    # No model HyperParameters

    # Training HyperParameters
    batch_size = 128,
    early_stopping_criteria = 5,
    learning_rate = 0.001,
    num_epoch = 10,
    seed = 1337
)

def make_train_state(args):
    return {
        'epoch_index': 0,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': -1,
        'test_acc': -1,
    }

train_state = make_train_state(args)

if not torch.cuda.is_available():
    args.cuda = False

args.device = torch.device("cuda" if args.cuda else "cpu")

# dataset and vectorizer
dataset = ReviewDataset.load_dataset_and_make_vectorize(args.review_csv)
vectorizer = dataset.get_vectorizer()

# model
classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
classifier = classifier.to(args.device)

# Loss and Optimizer
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

# the training loop
for epoch_index in range(args.num_epoch):

    print(f"\n Epoch {epoch_index+1}/{args.num_epoch}")
    train_state['epoch_index'] = epoch_index

    # Iterate  over training dataset
    # setup : batch generator, set loss and acc to 0, set train mode on
    dataset.set_split('train')
    total = len(dataset)
    batch_generator = generte_batches(dataset,
                                        batch_size=args.batch_size,
                                        device=args.device)

    running_loss = 0.0
    running_acc = 0.0
    classifier.train()

    for batch_index, batch_dict in enumerate(tqdm(batch_generator, total=total//args.batch_size)):

        # the training routine is 5 steps

        # step 1 zero gradients
        optimizer.zero_grad()

        # step 2. compute the output
        y_pred = classifier(x_in=batch_dict['x_data'].float(), apply_sigmoid=True)

        # step 3. compute the loss
        loss = loss_func(y_pred, batch_dict['y_label'].float())
        loss_batch = loss.item()
        running_loss += (loss_batch - running_loss) / (batch_index+1)

        # step 4. use loss to produce gradients
        loss.backward()

        # step 5. use optimizer to take gradient step
        optimizer.step()

        #------------------
        # compute the accuarcy
        acc_batch = accuracy_score(y_pred > 0.5, batch_dict['y_label'])
        running_acc += (acc_batch - running_acc) / (batch_index + 1)

    train_state['train_loss'].append(running_loss)
    train_state['train_acc'].append(running_acc)

    # iterate over val dataset

    # Iterate  over training dataset
    # setup : batch generator, set loss and acc to 0, set train mode on
    dataset.set_split('val')
    total = len(dataset)
    batch_generator = generte_batches(dataset,
                                      batch_size=args.batch_size,
                                      device=args.device)

    running_loss = 0.0
    running_acc = 0.0
    classifier.eval()

    print("\n >>>> Validation")
    for batch_index, batch_dict in enumerate(tqdm(batch_generator, total=total//args.batch_size)):

        # the training routine is 5 steps

        # step 1. compute the output
        y_pred = classifier(x_in=batch_dict['x_data'].float(), apply_sigmoid=True)

        # step 2. compute the loss
        loss = loss_func(y_pred, batch_dict['y_label'].float())
        loss_batch = loss.item()
        running_loss += (loss_batch - running_loss) / (batch_index + 1)

        # ------------------
        # compute the accuarcy
        acc_batch = accuracy_score(y_pred > 0.5, batch_dict['y_label'])
        running_acc += (acc_batch - running_acc) / (batch_index + 1)

    train_state['val_loss'].append(running_loss)
    train_state['val_acc'].append(running_acc)

# Test Set Evaluation
dataset.set_split('test')
total = len(dataset)
batch_generator = generte_batches(dataset,
                                  batch_size=args.batch_size,
                                  device=args.device)

running_loss = 0
running_acc = 0
classifier.eval()

print("\n >>> Testing")
for batch_index, batch_dict in enumerate(tqdm(batch_generator, total=total//args.batch_size)):

    # step 1. compute the output
    y_pred = classifier(x_in=batch_dict['x_data'].float(), apply_sigmoid=True)

    # step 2. compute the loss
    loss = loss_func(y_pred, batch_dict['y_label'].float())
    loss_batch = loss.item()
    running_loss += (loss_batch - running_loss) / (batch_index + 1)

    # ------------------
    # step 3. compute the accuarcy
    acc_batch = accuracy_score(y_pred > 0.5, batch_dict['y_label'])
    running_acc += (acc_batch - running_acc) / (batch_index + 1)

train_state['test_loss'] = running_loss
train_state['test_acc'] = running_acc

print("\n Test Loss: {}".format(train_state['test_loss']))
print("\n Test Acc: {}".format(train_state['test_acc']))

