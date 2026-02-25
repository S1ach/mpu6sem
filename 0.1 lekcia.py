import numpy as np
import torch

empty_tensor = torch.empty(3, 3)
print(empty_tensor)

tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
print(tensor_from_list)

tensor_from_list = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor_from_list)

numpy_array = np.array([6,7,8,9,10]);
tensor_from_numpy = torch.tensor(numpy_array);
print(tensor_from_numpy)

ones_tensor = torch.ones(2, 2)
print(ones_tensor)
zeros_tensor = torch.zeros(2, 2)
print(zeros_tensor)