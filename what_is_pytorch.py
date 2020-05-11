#-*- coding:utf-8 -*-
if __name__ == "__main__":
    import torch
    import numpy as np
    print("Let us start to learn...")
    # Construct a matrix
    x1 = torch.empty(5, 3)
    x2 = torch.rand(5, 3)
    x3 = torch.zeros(5, 5, dtype=torch.long)

    ###############
    # Construct a tensor directly from data
    ###############
    a1 = torch.tensor([1,2])
    a2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    temp = [[11, 22], [33, 44], [55, 66]]
    temp = np.array(temp)
    a3 = torch.tensor(temp)

    ###############
    # Create a tensor based on an existing tensor
    ###############
    x4 = x3.new_ones(5,5)
    # new意味着 x4与x3Tensor Attributes 相同(torch.dtype, torch.device, torch.layout等)
    # 再根据括号中的内容进行部分修改，如这里仅修改了size
    x = torch.rand_like(x4, dtype=torch.double)
    print(x.size())

    ###############
    # Operations
    ###############
    a2plus3 = a2 + a3
    a2plus3_ = torch.add(a2, a3)
    a2minus3 = a2 - a3
    a2pointwisemulti3 = a2*a3
    a3transpose = torch.transpose(a3, 0, 1)
    a3t = a3.t() # same as the one before
    a2matrixmulti3 = a2.mm(a3.t())

    # in-place
    a2.add_(a3)
    a2.copy_(a3)
    a2.t_()
    # indexing
    a2_1 = a2[:, -1]

    # view, reshape
    x = torch.tensor([[1, 2], [3, 4], [5, 6]])
    y = x.view(6)
    z = x.view(-1, 3)  # the size -1 is inferred from other dimensions
    print(x.size(), y.size(), z.size())

    # item
    # If you have a one element tensor, use.item() to get the value as a Python number
    xx = torch.tensor([2.8])
    print(xx)
    print(xx.item())

    ###############
    # With Numpy
    ###############
    # torch2np
    a2_np = a2.numpy()

    # np2tensor
    aa_np = np.ones(10)
    aa_tensor1 = torch.tensor(aa_np)
    aa_tensor2 = torch.from_numpy(aa_np)

    ###############
    # CUDA tensor
    ###############
    # let us run this cell only if CUDA is available
    # We will use ``torch.device`` objects to move tensors in and out of GPU
    if torch.cuda.is_available():
        print("cuda is available")
        device = torch.device("cuda")  # a CUDA device object
        y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
        x = x.to(device)  # or just use strings ``.to("cuda")``
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))  # ``.to`` can also change dtype together!



    print("the end.")