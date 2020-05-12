#-*- coding:utf-8 -*-
if __name__ == "__main__":
    import torch
    import numpy as np
    print("Let us start to learn...")
    # Central to all neural networks in PyTorch is the autograd package
    # The autograd package provides automatic differentiation for all operations on Tensors.
    # It is a define-by-run framework, which means that your backprop is defined by how your code is run,
    # and that every single iteration can be different.
    ################
    # Tensor requires_grad
    ###############
    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    z = y * y * 3
    z_mean = z.mean()
    # y, z, z_mean was created as a result of an operation, so it has a grad_fn(表示啥？有啥用？)

    # .requires_grad_( ... ) changes an existing Tensor’s requires_grad flag in-place.
    # The input flag defaults to False if not given.
    a = torch.randn(2, 2)
    print(a.requires_grad)
    a.requires_grad_(True)
    print(a.requires_grad)
    b = (a*a).sum()
    print(b.requires_grad)



    ################
    # Gradients
    ###############
    # Output is a scalar
    z_mean.backward()
    print(x.grad)

    # Output is a vector
    # Generally speaking, torch.autograd is an engine for computing vector-Jacobian product.
    # in the case y is no longer a scalar. torch.autograd could not compute the full Jacobian directly,
    # but if we just want the vector-Jacobian product, simply pass the vector to backward as argument:
    # 这里应该有某些理论上的近似原理，但输出为向量的情况应该不常见
    a = torch.randn(3, requires_grad=True)
    b = a * 2
    while b.data.norm() < 100:
        b = b * 2
    print(b)
    v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float) #关键步骤，给个向量！
    b.backward(v)
    print(a.grad)

    ###############
    # Stop Autograd
    ###############
    # to stop autograd from tracking history on Tensor (有啥用？没看懂)
    print("stop autograd...")
    print(x.requires_grad)
    print((x ** 2).requires_grad)
    with torch.no_grad():
        print((x ** 2).requires_grad)

    # to get a new Tensor with the same content but that does not require gradients by .detach()
    print(".detach()...")
    print(x.requires_grad)
    xx = x.detach()
    print(xx.requires_grad)
    print(x.eq(xx).all())



    print("the end.")