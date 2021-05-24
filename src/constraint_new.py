# Functions for computing the matrices A
# each function returns a matrix A = gamma[1] * A[1] \times \cdots \times gamma[i] A[i] and list of preconditionners gammas for ADMM

def matrixA1(N):
    A = np.zeros([N-1,N])
    for i in range(N-1):
        A[i][i] = -1
        A[i][i+1] = 1
    gamma1 = np.linalg.norm(A,2)
    A = gamma1 * A
    return A, [[N-1,gamma1]]

def matrixA1b(N):
    A = np.zeros([N,N])
    for i in range(N-1):
        A[i][i] = -1
        A[i][i+1] = 1
    A[N-1][N-1] = -1
    A[N-1][0] = 1
    gamma1 = np.linalg.norm(A,2)
    A = gamma1 * A
    return A, [[N,gamma1]]

def matrixA12(N):
    A1 = np.zeros([N-1,N])
    for i in range(N-1):
        A1[i][i] = -1
        A1[i][i+1] = 1
    A2 = np.transpose(A1).dot(A1)
    gamma1 = np.linalg.norm(A1,2)
    A1 = gamma1*A1
    gamma2 = np.linalg.norm(A2,2)
    A2 = gamma2*A2
    A = np.concatenate((A1,A2),axis = 0)
    return A, [[N-1,gamma1],[N,gamma2]]

def matrixA12b(N):
    A1 = np.zeros([N,N])
    for i in range(N-1):
        A1[i][i] = -1
        A1[i][i+1] = 1
    A1[N-1][N-1] = -1
    A1[N-1][0] = 1
    A2 = np.transpose(A1).dot(A1)
    gamma1 = np.linalg.norm(A1,2)
    A1 = gamma1*A1
    gamma2 = np.linalg.norm(A2,2)
    A2 = gamma2*A2
    A = np.concatenate((A1,A2),axis = 0)
    return A, [[N,gamma1], [N,gamma2]]

# PROJECTION TO Y

def ProjectToY(z, alphas):
    m = len(alphas)
    x = np.zeros([len(z),2])
    cnt = 0
    for i in range(m):
        c2 = alphas[i][1]**2
        for j in range(alphas[i][0]):
            d = z[cnt][0]**2 + z[cnt][1]**2
            if d > c2:
                den = math.sqrt(d)
                x[cnt] = alphas[i][1] * z[cnt] / den
            else:
                x[cnt] = z[cnt]
            cnt += 1
    return x

# ALGO WITHOUT LINEAR CONSTRAINTS B

def ProjectionNoB(z, x0, r0, A, U, V, alphas):
    x = np.copy(x0)
    r = np.copy(r0)
    eps = 0.001
    err = 1
    while err > eps:
        x_old = x
        y = ProjectToY(A.dot(x) + r,alphas)
        v = V.dot(y - r) + z
        x = np.linalg.solve(U,v)
        r = r + A.dot(x) - y
        err = np.linalg.norm(x - x_old)/np.linalg.norm(x)
    return x

def CurvingNoB(p0, pi, xgrid, ygrid, itermax, alphas, learning_rate = 0.1, circular = False): 
    N = len(p0)
    if circular:
        if len(alphas) == 1:
            A, gammas = matrixA1b(N)
        else:
            A, gammas = matrixA12b(N)
    else:
        if len(alphas) == 1:
            A, gammas = matrixA1(N)
        else:
            A, gammas = matrixA12(N)
    n = len(A)
    beta = 1 # need to be tuned for each application
    ATA = np.transpose(A).dot(A)
    U = beta * ATA + np.identity(N)
    V = beta * np.transpose(A)
    for i in range(len(gammas)):
        gammas[i][1] = alphas[i] * gammas[i][1]
    p = np.copy(p0)
    eps = 10**(-10)
    err = 2*eps
    iter = 0
    while err>eps and iter<itermax:
        if iter%50==0:
            plot_curves(p,iter)
        iter += 1
        grad = gradJ(p, pi, xgrid,ygrid)
        p = p - learning_rate * grad
        p = ProjectionNoB(p,np.zeros([N,2]),np.zeros([n,2]), A, U, V, gammas)
        err = np.linalg.norm(grad)
    return p

# ALGO WITH LINEAR CONSTRAINTS B

def Projection(z, x0, r0, A, U, V, b, alphas):
    N = len(x0)
    x = np.copy(x0)
    r = np.copy(r0)
    eps = 0.001
    err = 1
    while err > eps:
        x_old = x
        y = ProjectToY(A.dot(x) + r,alphas)
        v = np.concatenate((V.dot(y - r) + z,b),axis=0)
        x = np.linalg.solve(U,v)[:N] # solve the linear system
        r = r + A.dot(x) - y # update r
        err = np.linalg.norm(x - x_old)/np.linalg.norm(x) # compute error
    return x

def Curving(p0, pi, xgrid, ygrid, B, b, itermax, alphas, learning_rate = 0.1, circular = False): 
    N = len(p0)
    if circular:
        if len(alphas) == 1:
            A, gammas = matrixA1b(N)
        else:
            A, gammas = matrixA12b(N)
    else:
        if len(alphas) == 1:
            A, gammas = matrixA1(N)
        else:
            A, gammas = matrixA12(N)
    n = len(A)
    beta = 1 # need to be tuned for each application
    ATA = np.transpose(A).dot(A)
    U = beta * ATA + np.identity(N)
    U = np.concatenate((U,B),axis=0)
    U = np.concatenate((U,np.concatenate((np.transpose(B),np.zeros([len(B),len(B)])),axis=0)),axis=1)
    V = beta * np.transpose(A)
    for i in range(len(gammas)):
        gammas[i][1] = alphas[i] * gammas[i][1]
    
    p = np.copy(p0)
    eps = 10**(-10)
    err = 2*eps
    iter = 0
    
    while err>eps and iter<itermax:
        if iter%50==0:
            myplot_curve(p,iter)
        iter += 1
        grad = gradJ(p, pi, xgrid,ygrid)
        p = p - learning_rate * grad
        p = Projection(p,np.zeros([N,2]),np.zeros([n,2]), A, U, V, b, gammas)
        err = np.linalg.norm(grad)
    return p