import numpy as np
import matplotlib.pyplot as plt

fig_size = [18, 10]

sizes = np.array([4*4, 16*16, 32*32, 64*64, 128*128, 256*256, 512*512, 1024*1024, 2048*2048])
#
naive_times = np.array([0.001577, 0.0248, 0.560264, 1.07013, 7.2, 53.2, 443, 3914.04, 36224.6])
vinograd_times = np.array([0.000785, 0.021843, 0.163334, 0.944621, 10.2, 64.4, 576.8, 4263.55, 40736.5])
#
parallel_naive_times = np.array([22.94, 22.40, 23.19, 28.25, 63.51, 125.25, 1398.45, 9963, 10561.8])
parallel_vinograd_times = np.array([16.10, 77.80, 52.04, 58.88, 115.63, 67.45, 221.831, 825.65, 7575.5])

plt.figure(figsize=fig_size)
plt.title("Зависимость ускорения от размера матрицы")
#
plt.plot(sizes, naive_times, label = "naive_times")
plt.plot(sizes, vinograd_times, label = "vinograd_times")
# plt.scatter(sizes, naive_times)
#
plt.plot(sizes, parallel_naive_times, label = "parallel_naive_times")
plt.plot(sizes, parallel_vinograd_times, label = "parallel_vinograd_times")
#
plt.xlabel("Sizes")
plt.ylabel("Times, msecs")
plt.grid()
plt.legend()
plt.show()
