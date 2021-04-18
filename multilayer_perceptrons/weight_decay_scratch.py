import torch
from matplotlib import pyplot as plt
from _tools import mini_tool as tool
from _tools import mlp_frame as mlp


def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


def train(lambd):
    w, b = init_params()
    model, squared_loss = lambda X: mlp.linear_regression(X, w, b), mlp.squared_loss
    num_epochs, lr = 100, 0.003
    animator = tool.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs],
                             legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                loss = squared_loss(model(X), y) + lambd * l2_penalty(w)
            loss.sum().backward()
            mlp.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            print('epoch {}/{} begin...'.format(epoch + 1, num_epochs))
            animator.add(epoch + 1, (mlp.evaluate_loss(model, train_iter, squared_loss),
                                     mlp.evaluate_loss(model, test_iter, squared_loss)))
    print('L2 norm of w:', torch.norm(w).item())
    plt.show()


if __name__ == '__main__':
    num_train, num_test, num_inputs, batch_size = 20, 100, 200, 5
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    train_data = tool.synthetic_array(true_w, true_b, num_train)
    test_data = tool.synthetic_array(true_w, true_b, num_test)
    train_iter = tool.load_array(train_data, batch_size)
    test_iter = tool.load_array(test_data, batch_size, is_train=False)

    train(lambd=0)
    train(lambd=3)
