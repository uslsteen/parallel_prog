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

eps = 1e-3

def u(x, t):
    if t <= x:
        return t ** 2 - t * (t - x) + np.cos(np.pi * (t - x))
    elif t >= x:
        return t ** 2 - t * (t - x) + np.exp(x - t)
        #   
#
def comp_data_sets(data1, data2):
    for i in range(t_steps):
        for j in range(x_steps):
            if np.abs(data1[i][j] - data2[i][j]) > eps:
                #
                print("Error with results comparision")
                print(f"{data1[i][j]} - {data2[i][j]} > eps")

                return False
                #
#                
def get_analytic():
    analytic = np.zeros((t_steps, x_steps))

    for t_ind in range(t_steps):
        for x_ind in range(x_steps):
            analytic[t_ind][x_ind] = u(x[x_ind], t[t_ind])

    return analytic
    #
#
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
    #   
#
def plot(u, title):
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

    plt.title(title)
    plt.grid()
    plt.show()
    #
#
def main():
    analytic_solut = get_analytic()
    num_solut = get_data("data_parallel1.out")

    if not comp_data_sets(analytic_solut, num_solut):
        return

    plot(num_solut, "График численного решения")
    plot(num_solut, "График аналитического решения")
     
    

if __name__ == '__main__':
    main()