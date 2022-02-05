# 该部分记录的是pytorch的核心概念理解

## 1： 动态计算图

### 一： 动态图的简介

Pytorch的计算图由节点和边组成，节点表示张量或者Function，边表示张量和Function之间的依赖关系。

Pytorch的计算图是动态图，其含义是：
计算图的正向传播是立即执行的。无需等待完整的计算图创建完毕，每条语句都会在计算图中动态添加节点和边，并立即执行正向传播得到计算结果。
计算图在反向传播后立即销毁。下次调用时需要重新构建计算图。如果在程序中使用了backward方法执行力反向传播，或者理由torch.autograd.grad方法计算了梯度，那么创建的计算图会被立即销毁，释放存储空间，下次调用需要重新创建。
    
    import torch 
    w = torch.tensor([[3.0,1.0]],requires_grad=True)
    b = torch.tensor([[3.0]],requires_grad=True)
    X = torch.randn(10,2)
    Y = torch.randn(10,1)
    Y_hat = X@w.t() + b  
    loss = torch.mean(torch.pow(Y_hat-Y,2))
Y_hat定义后其正向传播被立即执行，与其后面的loss创建语句无关