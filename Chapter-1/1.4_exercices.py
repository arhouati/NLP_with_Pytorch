import torch

# Helper function to describe tensors
def describe(x) -> None:
    print("Type : ", x.type())
    print("Shape/size : ", x.shape)
    print("Values : ", x)

# 1. Create a 2D tensor and Then add a dimension of size 1 inserted at dimension 0
x = torch.rand(2, 2)
describe(x)
x.unsqueeze(0)
describe(x)

# 2. Remove the extra dimension you just added to the previous tensor
x.squeeze(0)

# 3. create a random tensor of shape 5x3 in the interval [3, 7)
x = torch.randint(3, 7, [5, 3])
describe(x)

# 4. Create a tensor with values from a normal distribution (mean = 0, std = 1)
x = torch.rand(3, 3)
x.normal_()
describe(x)

# 5. Retrieve the indexes of all nonzero elements in the tensor torch.Tensor([1, 1, 1, 0, 1])
x = torch.Tensor([1, 1, 1, 0, 1])
describe(torch.nonzero(x))

# 6. Create a random tensor of size (3, 1) and then horizontally stacj four copies together
x = torch.rand(3, 1)
describe(x.expand(3, 4))

# 7. Return the batch matrix-matric product of two three-dimensional matrices (a=torch.rand(3, 4, 5), b=torch.rand(3, 5,4))
a=torch.rand(3, 4, 5)
b=torch.rand(3, 5, 4)
describe(a)
describe(b)

describe(torch.bmm(a, b))

# 8. Return the batch matrix-matrix product of 3D matrix and 2D matrix (a=torch.rand(3, 4, 5), b=torch.rand(5,4))
a=torch.rand(3, 4, 5)
b=torch.rand(5, 4)
c = torch.bmm(a, b.unsqueeze(0).expand(a.size(0), *b.size))
describe(c)