import torch.nn as nn
import torch

class ReviewClassifier(nn.Module):
    """
    a simple perceptron-based classifer
    """
    def __init__(self, num_features):
        """

        :param num_features (int): the size ot the input feature vector
        """
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features,
                             out_features=1)


    def forward(self, x_in, apply_sigmoid: bool=False):
        """ The forward pass of the classifier

        :param x_ini (torch.Tensor): an input data tensor. x_ini.shape should be (Batch, num_features)
        :param apply_sigmoid: a flag for the sigmoid activation, should be false if used with cross-entropy loss function
        :return: thre resulting tensor. tensor.shape should be (batch,).
        """
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)

        return y_out



