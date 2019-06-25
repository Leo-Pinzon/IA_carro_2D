import csv
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as plticker

diretorio = "data/grafico_T4-20.txt"

scores = []
ultimos = []
medias = []
tamUltimos = 40
aux = 1
counter = 0

with open(diretorio, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if aux > 0:
            scores.append(float(row[0]))
            ultimos.append(float(row[0]))
            if ultimos.__len__() > tamUltimos:
                ultimos.pop(0)
            medias.append(np.mean(ultimos))
        aux = aux * -1

fig, ax = plt.subplots()
ax.plot(scores,'g.')
ax.plot(medias,'k')
plt.grid(True)
xa = plticker.MultipleLocator(base=50)
ax.xaxis.set_major_locator(xa)
plt.yticks(np.arange(0,250,10),np.arange(0,250,10))


plt.show()