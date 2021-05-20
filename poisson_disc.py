import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dist(p1,p2):
    p = [p1[0]-p2[0],p1[1]-p2[1]]
    return p[0]**2 + p[1]**2

def randPoint(p, r):
    l = random.uniform(r,2*r)
    a = random.random()*2*math.pi
    p2 = [l*math.cos(a)+p[0], l*math.sin(a)+p[1]]
    while p2[0] > 1 or p2[0] < 0 or p2[1] > 1 or p2[1] < 0:
        l = random.uniform(r,2*r)
        a = random.random()*2*math.pi
        p2 = [l*math.cos(a)+p[0], l*math.sin(a)+p[1]]
    return p2

def placeGrid(p, index, grid):
    N = len(grid)-2
    i = math.floor(p[0]*N)
    j = math.floor(p[1]*N)
    grid[i+1][j+1] = index

def neighbor_list(i,j,grid):
    return [grid[i][j+1],grid[i][j-1],grid[i+1][j+1],grid[i+1][j],grid[i+1][j-1],grid[i-1][j+1],grid[i-1][j],grid[i-1][j-1]]

def PoissonDisc(r,maxiter):
    r2 = r**2
    N = math.ceil(math.sqrt(2)/r) # here
    grid = np.zeros([N+2, N+2],dtype = int)
    for i in range(N+2):
        for j in range(N+2):
            grid[i][j] = -1

    samples=[[random.random(), random.random()]]
    active_list = [0]
    placeGrid(samples[0],0,grid)
    index = 1
    while active_list:
        it = random.randint(0,len(active_list)-1)
        p = samples[active_list[it]]
        for iter in range(maxiter):
            p2 = randPoint(p,r)
            flag = True
            i = math.floor(p2[0]*N)+1
            j = math.floor(p2[1]*N)+1
            if grid[i][j] != -1:
                continue
            neighbors = neighbor_list(i,j,grid)
            for neighbor in neighbors:
                if neighbor != -1 and dist(p2, samples[neighbor]) < r2:
                    flag = False
                    break
            if flag:
                active_list.append(index)
                samples.append(p2)
                placeGrid(p2,index,grid)
                index += 1
        if iter >= maxiter-1:
            del active_list[it]
    return samples

def scale(samples):
    rescale = []
    for i in range(len(samples)):
        rescale.append([samples[i][0]-0.5,samples[i][1]-0.5])
    return np.array(rescale)