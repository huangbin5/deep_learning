import numpy as np
import networkx
from networkx import to_numpy_matrix
from matplotlib import pyplot as plt


def relu(M):
    # for i in range(M.shape[0]):
    #     for j in range(M.shape[1]):
    #         if M[i, j] < 0:
    #             M[i, j] = 0
    return M


def example():
    A = np.matrix([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 0]],
        dtype=float
    )
    X = np.matrix([
        [i, -i]
        for i in range(A.shape[0])
    ], dtype=float)

    I = np.matrix(np.eye(A.shape[0]))
    A_hat = A + I

    D_hat = np.array(np.sum(A_hat, axis=0)[0])[0]
    D_hat = np.matrix(np.diag(D_hat))

    W = np.matrix([
        [1, -1],
        [-1, 1]
    ])

    print(relu(D_hat.I @ A_hat @ X @ W))


def zachary():
    G = networkx.karate_club_graph()
    nodes = sorted(list(G.nodes))
    A = to_numpy_matrix(G, nodelist=nodes)
    I = np.eye(G.number_of_nodes())
    X = I
    A_hat = A + I
    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = np.matrix(np.diag(D_hat))
    W_1 = np.random.normal(loc=0, scale=1, size=(G.number_of_nodes(), 4))
    W_2 = np.random.normal(loc=0, size=(W_1.shape[1], 3))
    W_3 = np.random.normal(loc=0, size=(W_2.shape[1], 2))

    H_1 = relu(D_hat.I @ A_hat @ X @ W_1)
    H_2 = relu(D_hat.I @ A_hat @ H_1 @ W_2)
    output = relu(D_hat.I @ A_hat @ H_2 @ W_3)
    output = np.array(output)

    x, y = output[:, 0], output[:, 1]
    color = ['r' if G.nodes[i]['club'] == 'Officer' else 'b' for i in G]
    plt.scatter(x, y, c=color)
    plt.waitforbuttonpress()


if __name__ == '__main__':
    zachary()
