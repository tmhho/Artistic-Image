def matrixA1(N):
    A = np.zeros([N-1,N])
    for i in range(N-1):
        A[i][i] = -1
        A[i][i+1] = 1
    l1 = np.linalg.norm(A,2)
    A = l1*A
    ATA = np.transpose(A).dot(A)
    return A, ATA

def matrixA1_circ(N):
    A = np.zeros([N,N])
    for i in range(N-1):
        A[i][i] = -1
        A[i][i+1] = 1
    A[N-1][N-1] = -1
    A[N-1][0] = 1
    l1 = np.linalg.norm(A,2)
    A = l1*A
    ATA = np.transpose(A).dot(A)
    return A, ATA

def matrixA1A2(N):
    A1 = np.zeros([N-1,N])
    for i in range(N-1):
        A1[i][i] = -1
        A1[i][i+1] = 1
    A2 = np.transpose(A1).dot(A1)

    l1 = np.linalg.norm(A1,2)
    A1 = l1*A1
    l2 = np.linalg.norm(A2,2)
    A2 = l2*A2

    A = np.concatenate((A1,A2),axis = 0)
    ATA = np.transpose(A).dot(A)
    return A, ATA

def matrixA1A2_circ(N):
    A1 = np.zeros([N,N])
    for i in range(N-1):
        A1[i][i] = -1
        A1[i][i+1] = 1
    A1[N-1][N-1] = -1
    A1[N-1][0] = 1
    A2 = np.transpose(A1).dot(A1)

    l1 = np.linalg.norm(A1,2)
    A1 = l1*A1
    l2 = np.linalg.norm(A2,2)
    A2 = l2*A2
    
    A = np.concatenate((A1,A2),axis = 0)
    ATA = np.transpose(A).dot(A)
    return A, ATA

def ProjectToY(Z, c, N):
    Y = np.zeros([len(Z),2])
    c2 = c**2
    for i in range(len(Z)):
        d = Z[i][0]**2 + Z[i][1]**2
        if d > c2:
            den = math.sqrt(d)
            Y[i] = c * Z[i] / den
        else:
            Y[i] = Z[i]
    return Y

def ProjectToY2(Z, c, N):
    m = len(c)    
    Y = np.zeros([N,2])
    cnt = 0
    for i in range(m):
        c2 = c[i]**2
        for j in range(len(Z[i])):
            d = Z[i][j][0]**2 + Z[i][j][1]**2
            if d > c2:
                den = math.sqrt(d)
                Y[cnt] = c[i] * Z[i][j] / den
            else:
                Y[cnt] = Z[i][j]
            cnt += 1
    return Y

def Projector(z, x0, r0, A, ATA, c, beta = 1):
    N = len(x0)
    x = np.copy(x0)
    r = np.copy(r0)
    eps = 0.001
    err = 1
    while err > eps:
        tmp = A.dot(x) + r
        y = ProjectToY(tmp,c,len(tmp))
        U = beta * ATA + np.identity(N)
        b = beta * np.transpose(A).dot(y - r) + z
        x_old = x
        x = np.linalg.solve(U,b)
        r = r + A.dot(x) - y
        err = np.linalg.norm(x - x_old)/np.linalg.norm(x)
    return x

def Projector2(z, x0, r0, A, ATA, c, beta = 1):
    N = len(x0)
    x = np.copy(x0)
    r = np.copy(r0)
    eps = 0.001
    err = 1
    while err > eps:
        tmp = A.dot(x) + r
        y = ProjectToY2([tmp[:-N],tmp[-N:]],c,len(tmp))
        U = beta * ATA + np.identity(N)
        b = beta * np.transpose(A).dot(y - r) + z
        x_old = x
        x = np.linalg.solve(U,b)
        r = r + A.dot(x) - y
        err = np.linalg.norm(x - x_old)/np.linalg.norm(x)
    return x

def ProjectorFull(z, x0, r0, A, ATA, alpha, B, b, beta = 1):
    N = len(x0)
    x = np.copy(x0)
    r = np.copy(r0)
    eps = 0.001
    err = 1
    while err > eps:
        tmp = A.dot(x) + r
        y = ProjectToY2([tmp[:-N],tmp[-N:]],alpha,len(tmp))
        U = beta * ATA + np.identity(N)
        b = beta * np.transpose(A).dot(y - r) + z
        x_old = x
        x = np.linalg.solve(U,b)
        x = x[:N]
        r = r + A.dot(x) - y
        err = np.linalg.norm(x - x_old)/np.linalg.norm(x)
    return x


def GradDescent_constraint(J, gradJ, p0, pi, xgrid, ygrid, itermax, constraint, learning_rate = 0.1): 
    N = len(p0)
    A, ATA = matrixA1(N)
    p = np.copy(p0)
    eps = 1e-10
    err = 2*eps
    iter = 0
    beta = 1
    myplot_curve(p,iter)
    while err>eps and iter<itermax:
        grad = gradJ(p, pi, xgrid,ygrid)
        p = p - learning_rate*grad
        p = Projector(p,np.zeros([N,2]),np.zeros([len(A),2]), A, ATA, constraint, beta)
        err = np.linalg.norm(grad)
        iter += 1
        if iter%50==0:
            myplot_curve(p,iter)
    return p