import torch
from torch import nn
from matplotlib import pyplot as plt
from _tools import mini_tool as tool
from _tools import mlp_frame as mlp


def train(lambd):
    model = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in model.parameters():
        param.data.normal_()
    mse_loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    # bias不指定weight_decay
    optimizer = torch.optim.SGD([{"params": model[0].weight, 'weight_decay': lambd}, {"params": model[0].bias}], lr=lr)
    animator = tool.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs],
                             legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                optimizer.zero_grad()
                loss = mse_loss(model(X), y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print('epoch {}/{} begin...'.format(epoch + 1, num_epochs))
            animator.add(epoch + 1, (mlp.evaluate_loss(model, train_iter, mse_loss),
                                     mlp.evaluate_loss(model, test_iter, mse_loss)))
    print('L2 norm of w:', model[0].weight.norm().item())
    plt.show()


if __name__ == '__main__':
    num_train, num_test, num_inputs, batch_size = 20, 100, 200, 5
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    train_data = tool.synthetic_array(true_w, true_b, num_train)
    test_data = tool.synthetic_array(true_w, true_b, num_test)
    train_iter = tool.load_array(train_data, batch_size)
    test_iter = tool.load_array(test_data, batch_size, is_train=False)

    train(0)
    train(3)
