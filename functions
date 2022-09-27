import inspect
import time
from scipy import linalg, sparse
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import eye
from scipy.sparse import linalg

def report(xk):
    frame = inspect.currentframe().f_back
    print(frame.f_locals['resid'])

def precon(A,n,c):
    start_time = time.time()
    print('Initiating Preconditioner')
    A_new = A+ c*sparse.eye(n,n);
    #return A_new
    ilu = splu(A_new)
    #return ilu
    Mx = lambda x: ilu.solve(x)
    #return Mx
    M = LinearOperator((n, n), Mx)
    print("---Preconditioning took %s seconds ---" % (time.time() - start_time))
    return M

def bicgstab(A,b,M):
    start_time = time.time()
    print('Initiating Linear Solver')
    x, exit_code = linalg.bicgstab(A, b, x0=None, tol=1e-16, maxiter=1000, M=M) #,callback=report)
    if exit_code == 0:
        print("Converged")
    else:
        print("Not converged")
    print("--- linear solver took %s seconds ---" % (time.time() - start_time))
    return x
