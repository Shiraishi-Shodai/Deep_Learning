#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

#sns.set()
#sns.set_style('whitegrid')
#sns.set_palette('gray')

fig = plt.figure()


########### 恒等関数
ax1 = fig.add_subplot(2, 2, 1)
x = np.linspace(-10,10,200)
y1 = x
ax1.plot(x, y1)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("恒等関数")

###########
ax2 = fig.add_subplot(2, 2, 2)
y1 = np.maximum(0, x)
ax2.plot(x, y1)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("ReLU")


###########
ax3 = fig.add_subplot(2, 2, 3)
y3 = 1/(1+np.exp(-x))
ax3.plot(x, y3)
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_title("シグモイド")

###########
ax4 = fig.add_subplot(2, 2, 4)
y4 = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
ax4.plot(x, y4)
ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_title("ハイパボリックタンジェント")

# show plots
#fig.tight_layout()
fig.show()
