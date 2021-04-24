import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

def H(g,eps=1e-4):
    return np.zeros(g.shape[:-1])
def Hp(g,eps=1e-4):
    return np.zeros_like(g)

def convolution(pi, grid):
    return np.zeros_like(pi)
def convolution_prime(pi, grid):
    return np.zeros(pi.shape+(2,))

def interp(grid, val, pos):
    return np.zeros(pos.shape[0])
def interp_prime(grid, val, pos):
    return np.zeros_like(pos)


def F(p):
    return 42
def gradF(p):
    grad = np.zeros_like(p)
    return grad

def G(pi, grid, p):
    return 0
def gradG(pi, grid, p):
    return np.zeros_like(p)


if __name__=='__main__':
    N = 65
    M = 64
    grid = (np.arange(N)+0.5)/N - 0.5
    grid_x, grid_y = np.meshgrid(grid, grid)

    pi = np.exp(-(grid_x**2+grid_y**2)/0.05)
    pi /= np.sum(pi)
    # plt.imshow(pi)
    # plt.show()

    Niter = 200
    step = 1e1
    p = np.random.uniform(-0.25,0.25,(M,2))
    for nit in range(Niter):
        gF = gradF(p)
        gG = gradG(pi, grid, p)
        grad = gG-gF
        p = p-step*grad
        loss = G(pi, grid, p)-F(p)

        if nit%10==0:
            plt.figure(1)
            plt.clf()
            plt.imshow(pi)
            plt.pause(1e-3)
            plt.figure(2)
            plt.clf()
            plt.scatter(p[:,0],p[:,1],s=4)
            plt.title("nit="+str(nit))
            plt.pause(1e-3)
