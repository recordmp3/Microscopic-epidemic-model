import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# figures drawn in python for debugging
def ave_(a, lim=1):
    b = []
    for i in range(len(a)):
        b.append(ave(a[i], lim))
    return b

def ave(a, lim=1):
    b = [] 
    su = 0
    for i in range(len(a)):
        su += a[i]
        if i >= lim: su -= a[i - lim]
        b.append(su / min(lim, i + 1))
    return b

class Draw():

    def init(self):

        plt.style.use('ggplot')
        plt.ion()
        self.fig = plt.figure(figsize=(4 * (self.shape % 10), 4 * (self.shape // 10)))
        self.ax = [self.fig.add_subplot(self.shape % 10, self.shape // 10, 1 + i) for i in range(self.n)]

    def __init__(self, shape, save_path="/"):

        self.save_path = save_path
        self.shape = shape
        self.n = self.shape % 10 * (self.shape // 10)
        
    def reset(self):
        
        plt.clf()
        self.ax = [self.fig.add_subplot(self.shape % 10, self.shape // 10, 1 + i) for i in range(self.n)]

    def save(self, title):

        plt.savefig(os.path.join(self.save_path, title + '.jpg'),
                    dpi=120)  # bbox_inches='tight') # bbox_inches='tight')

    def __del__(self):

        self.close()

    def close(self):

        plt.clf()
        plt.cla()
        plt.close()

    def show(self):

        plt.show()
        plt.pause(0.001)

    def draw(self, x, y, x_label, y_label, title, pos, label=None, one_x=True, y_d=None, y_u=None):

        if label is None: label = [None for i in range(len(y))]
        for i in range(len(y)):
            if one_x:
                self.ax[pos].plot(x, y[i], linewidth=1, label=label[i])
                if y_d is not None:
                    self.ax[pos].fill_between(x, y_d[i], y_u[i], alpha=0.4)
            else:
                self.ax[pos].plot(x[i], y[i], linewidth=1, label=label[i])
                if y_d is not None:
                    self.ax[pos].fill_between(x[i], y_d[i], y_u[i], alpha=0.4)
        self.ax[pos].set_xlabel(x_label)  # , fontsize=15)
        self.ax[pos].set_ylabel(y_label)  # , fontsize=15)
        self.ax[pos].set_title(title)
        self.ax[pos].legend(numpoints=1, fancybox=True)  # fontsize=15)
        # self.ax[pos].figure.canvas.draw()

def draw(x, y, x_label, y_label, title, save_path, label = None, one_x = True, y_d = None, y_u = None, color_use = True):
    color_list = ['crimson', 'orangered', 'pink', 'yellow', 'green', 'blue', 'purple', 'lightsalmon', 'maroon', 'black', 'grey', 'orchid', 'fuchsia', 'steelblue'] * 10
    plt.style.use('ggplot')
    print('save', save_path)
    if label is None:label = [None for i in range(len(y))]
    for i in range(len(y)):
        if one_x:
            plt.plot(x, y[i], linewidth=1, label = label[i], color = color_list[i])
            if y_d is not None:
                plt.fill_between(x, y_d[i], y_u[i], alpha=0.4)
        else:
            plt.plot(x[i], y[i], linewidth=1, label = label[i], color = color_list[i])
            if y_d is not None:
                plt.fill_between(x[i], y_d[i], y_u[i], alpha=0.4)
    plt.xlabel(x_label)#, fontsize=15)
    plt.ylabel(y_label)#, fontsize=15)
    plt.title(title)
    plt.legend(numpoints=1, fancybox=True)# fontsize=15)
    plt.savefig(save_path + '/' + title + '.png', dpi=120)# bbox_inches='tight') # bbox_inches='tight')
    #plt.show()
    plt.close()

def generate_graph(g_i, save_path, step):

    np.save(os.path.join(save_path, "g_i_" + str(step)+".npy"), np.array([g_i]))
