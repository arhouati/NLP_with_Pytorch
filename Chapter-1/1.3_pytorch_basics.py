import torch
import numpy as np

# Helper function to describe tensors
def describe(x) -> None:
    print("Type : ", x.type())
    print("Shape/size : ", x.shape)
    print("Values : ", x)

print("#"*50)
print("#1 : creating a tensor by specifying its dimensiosn")
describe(torch.Tensor(2, 3))

print("#"*50)
print("#2 : creating a randomly initialized tensor")
describe(torch.rand(2,3)) # uniform random
describe(torch.randn(2,3)) # random normal

print("#"*50)
print("#3 : creating a filled tensor")
describe(torch.zeros(2,3))
x = torch.ones(2,3)
describe(x)
x.fill_(5)
describe(x)

print("#"*50)
print("#4 : creating and initializing tensor from lists")
x = torch.Tensor([[1,2,3],
                  [4,5,6]])
describe(x)

print("#"*50)
print("#5 : creating and initializing tensor from NumPy")
npy = np.random.rand(2, 3)
x = torch.from_numpy(npy)
describe(x)

print("#"*50)
print("#6 : Tensor types and size")
x = torch.Tensor(2, 3) # a default tensor is Float
describe(x)
x = torch.FloatTensor(2,3) # a float tensor
describe(x)
x = torch.LongTensor(2,3) #  a long tensor
describe(x)
x = x.float() # a casting of the tensor (we can also use .long())
describe(x)
x = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.int64) # tensor type using dtype
describe(x)
x = x.float()
describe(x)

print("#"*50)
print("#6 : Tensor operations")
x = torch.randn(2, 3)
describe(x)

# arithmetic operations
describe(torch.add(x, x))
describe(x + x)

# dimension's operations
x = torch.arange(6)
describe(x)

x = x.view(2, 3)
describe(x)

describe(torch.sum(x, dim=0))
describe(torch.sum(x, dim=1))
describe(torch.transpose(x, 0, 1))

print("#"*50)
print("#7 : Indexing, Slicing and Joining")
x = torch.arange(6).view(2, 3)
describe(x)

describe(x[:1, :2])
describe(x[0, 1])

indices = torch.LongTensor([0, 2])
describe(torch.index_select(x, dim=1, index=indices))

indices = torch.LongTensor([0, 0])
describe(torch.index_select(x, dim=0, index=indices))

row_indices = torch.arange(2).long()
col_indices = torch.LongTensor([0, 1])
describe(x[row_indices, col_indices])

# concatenation of tensors
x = torch.arange(6).view(2, 3)
describe(x)

describe(torch.cat([x, x], dim=0))
describe(torch.cat([x, x], dim=1))
describe(torch.stack([x,x]))

# linear algebra
x1 = torch.arange(6).view(2, 3).float()
describe(x1)

x2 = torch.ones(3, 2)
x2[:, 1] += 1
describe(x2)

x3 = torch.mm(x1, x2)
describe(x3)

# computational graphs
x= torch.ones(2, 2, requires_grad=True)
describe(x)
print(x.grad is None)

y = (x + 2) * (x + 5) + 3
describe(y)
print(x.grad is None)

z = y.mean()
describe(z)
z.backward()
print(x.grad is None)

print("#"*50)
print("#8 : CUDA Tensors (using GPU) ")
print(torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x = torch.rand(3, 3).to(device)
describe(x)

