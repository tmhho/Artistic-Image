import buggyscript as bs
import numpy as np


def check_H(H, Hp, eps=1e-8):
    x = np.random.randn(10,10,2)
    y = H(x, eps=eps).sum()
    g = Hp(x, eps=eps)
    gn = np.zeros_like(g)
    h = 1e-6
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                xh = x.copy(); xh[i,j,k]+=h
                yh = H(xh, eps=eps).sum()
                gn[i,j,k] = (yh-y)/h
    err = np.linalg.norm(gn-g)/np.linalg.norm(g)
    if err<1e-5:
        print(" H and Hp OK ".center(30,'#'))
    else:
        raise Exception("H and Hp are not consistent. Please check that H and Hp are properly implemented. Error=%3f"%err)

def check_interp(interp):
    N = 21
    grid = (np.arange(N)+0.5)/N - 0.5
    grid_x, grid_y = np.meshgrid(grid, grid)

    pos = np.zeros((N,N,2))
    pos[...,0] = grid_x
    pos[...,1] = grid_y
    pos = pos.reshape(-1,2)
    val = np.random.randn(N,N)
    I = interp(grid, val, pos).reshape(N,N).T
    err1 = np.linalg.norm(I-val)/np.linalg.norm(val)

    pos = np.zeros((N,N-1,2))
    pos[...,0] = (grid_x[:,1:]+grid_x[:,:-1])/2
    pos[...,1] = grid_y[:,:-1]
    pos = pos.reshape(-1,2)
    val = np.random.randn(N,N)
    I = interp(grid, val, pos).reshape(N,N-1).T
    val2 = (val[1:]+val[:-1])/2
    err2 = np.linalg.norm(I-val2)/np.linalg.norm(val2)

    pos = np.zeros((N-1,N,2))
    pos[...,0] = grid_x[:-1]
    pos[...,1] = (grid_y[1:]+grid_y[:-1])/2
    pos = pos.reshape(-1,2)
    val = np.random.randn(N,N)
    I = interp(grid, val, pos).reshape(N-1,N).T
    val2 = (val[:,1:]+val[:,:-1])/2
    err3 = np.linalg.norm(I-val2)/np.linalg.norm(val2)

    if max(err1, err2, err3)<1e-12:
        print(" interp OK ".center(30,'#'))
    else:
        raise Exception("interp is not properly implemented")


def check_interp_prime(interp_prime):
    N = 21
    grid = (np.arange(N)+0.5)/N - 0.5
    grid_x, grid_y = np.meshgrid(grid, grid)

    pos = np.zeros((N,N,2))
    pos[...,0] = grid_x
    pos[...,1] = grid_y
    pos = pos.reshape(-1,2)
    val = np.random.randn(N,N,2)
    I = interp_prime(grid, val, pos).reshape(N,N,2)
    I[...,0], I[...,1] = I[...,1].T.copy(), I[...,0].T.copy()
    err1 = np.linalg.norm(I-val)/np.linalg.norm(val)

    pos = np.zeros((N,N-1,2))
    pos[...,0] = (grid_x[:,1:]+grid_x[:,:-1])/2
    pos[...,1] = grid_y[:,:-1]
    pos = pos.reshape(-1,2)
    val = np.random.randn(N,N,2)
    I = interp_prime(grid, val, pos).reshape(N,N-1,2)
    val2 = (val[1:]+val[:-1])/2
    I2 = np.zeros_like(val2)
    I2[...,0], I2[...,1] = I[...,1].T.copy(), I[...,0].T.copy()
    err2 = np.linalg.norm(I2-val2)/np.linalg.norm(val2)

    pos = np.zeros((N-1,N,2))
    pos[...,0] = grid_x[:-1]
    pos[...,1] = (grid_y[1:]+grid_y[:-1])/2
    pos = pos.reshape(-1,2)
    val = np.random.randn(N,N,2)
    I = interp_prime(grid, val, pos).reshape(N-1,N,2)
    val2 = (val[:,1:]+val[:,:-1])/2
    I2 = np.zeros_like(val2)
    I2[...,0], I2[...,1] = I[...,1].T.copy(), I[...,0].T.copy()
    err3 = np.linalg.norm(I2-val2)/np.linalg.norm(val2)

    if max(err1, err2, err3)<1e-12:
        print(" interp_prime OK ".center(30,'#'))
    else:
        raise Exception("interp_prime is not properly implemented")

def check_F(F, gradF):
    x = np.random.randn(20,2)
    y = F(x)
    g = gradF(x)
    gn = np.zeros_like(g)
    h = 1e-6
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            xh = x.copy(); xh[i,j]+=h
            yh = F(xh)
            gn[i,j] = (yh-y)/h
    err = np.linalg.norm(gn-g)/np.linalg.norm(g)
    if err<1e-5:
        print(" F and gradF OK ".center(30,'#'))
    else:
        raise Exception("F and gradF are not consistent. Please check that F and gradF are properly implemented. Error=%3f"%err)


if __name__=='__main__':
    check_H(bs.H, bs.Hp)
    check_F(bs.F, bs.gradF)
    check_interp(bs.interp)
    check_interp_prime(bs.interp_prime)
