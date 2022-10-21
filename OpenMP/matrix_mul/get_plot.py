import numpy as np
import matplotlib.pyplot as plt

fig_size = [18, 10]

sizes = np.array([4*4, 16*16, 32*32, 64*64, 128*128, 256*256, 512*512, 1024*1024, 2048*2048])
naive_times = np.array([])
vinograd_times = np.array([])

plt.figure(figsize=fig_size)
plt.title("Зависимость ускорения от размера матрицы")

plt.plot(sizes, naive_times)
plt.plot(sizes, vinograd_times)

plt.xlabel("p")
plt.ylabel("S")
plt.grid()
plt.show()