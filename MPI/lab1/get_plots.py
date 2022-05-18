import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

tau = 1e-2
h = 1e-2

T = X = 1

x_steps = int(X / h)
t_steps = int(T / tau)

x = np.arange(start=0, stop=X, step=h)
t = np.arange(start=0, stop=T, step=tau)


def get_data(src):
    data = np.zeros((t_steps, x_steps))

    res_list = list()

    with open(src, 'r') as f:
        for line in f:
            res_list.append(float(line.split()[0]))
            
    for i in range(t_steps):
        for j in range(x_steps):
            data[i][j] = res_list[i * x_steps + j]
            
    return data

def plot(u):
    x = np.arange(start=0, stop=X, step=h)
    t = np.arange(start=0, stop=T, step=tau)
    
    x, t = np.meshgrid(x, t)

    fig = plt.figure(figsize=(16, 16))
    graph = plt.axes(projection='3d')

    surf = graph.plot_surface(x, t, u, cmap=cm.plasma)
    graph.set_xlabel("x", fontsize=20)
    graph.set_ylabel("t", fontsize=20)

    graph.set_zlabel("u(x, t)", fontsize=20)

    fig.colorbar(surf, shrink=0.5)

    plt.grid()
    plt.show()


def main():
    plot(get_data("data_parallel.out"))


if __name__ == '__main__':
    main()