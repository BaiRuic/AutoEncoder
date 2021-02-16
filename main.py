import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from ae import AutoEncoder
from vae import Vari_AutoEncoder
import visdom

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  

def main():

    # 加载数据
    mnist_train = datasets.MNIST(root='data', train=True, transform=transforms.Compose(
        [transforms.ToTensor()]
    ), download=True)
    mnist_train = DataLoader(dataset=mnist_train, batch_size=32, shuffle=True)

    mnist_test = datasets.MNIST(root='data', train=True, transform=transforms.Compose(
        [transforms.ToTensor()]
    ), download=True)
    mnist_test = DataLoader(dataset=mnist_test, batch_size=32, shuffle=True)

    # 定义模型，损失函数，优化器
    # model = AutoEncoder()
    model = Vari_AutoEncoder()
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 实例化 visdom 图示
    viz = visdom.Visdom()
    # 训练
    for epoch in range(1000):
        for batchidx, (x, _) in enumerate(mnist_test):
            # x.shape : [batch_size, 1, 28, 28]
            x.to(DEVICE) 
            # forward
            x_hat, kld = model(x)
            loss = criterion(x_hat, x)
            if kld is not None:
                elbo = -loss -kld
                loss = -elbo
            
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'epoch:{epoch},loss{loss.item()}, kld:{kld}')
        
        # 评估模型
        
        x, _ = iter(mnist_test).next()
        x.to(DEVICE)
        with torch.no_grad():
            x_hat, kld = model(x)
        # 一个batch是32 所以一行放八个，总共四行
        viz.images(x, nrow=8, win='x', opts=dict(title='x'))
        viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))



if __name__ == '__main__':
    main()  