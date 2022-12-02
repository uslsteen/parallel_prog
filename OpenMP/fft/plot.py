import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fig_size = [18, 10]

data = pd.read_csv('data.csv')
sizes = data["Size"] 
names = list()

for it in data:
    names.append(it)

plt.figure(figsize=fig_size)
plt.title("Зависимость ускорения количества сигналов")
#
# plt.plot(sizes, data[names[1]], label = names[1])
plt.plot(sizes, data[names[2]], label = names[2])
plt.plot(sizes, data[names[3]], label = names[3])
plt.plot(sizes, data[names[4]], label = names[4])
plt.plot(sizes, data[names[5]], label = names[5])
#
plt.xlabel("Sizes")
plt.ylabel("Times, microsecs")
plt.grid()
plt.legend()
plt.show()
