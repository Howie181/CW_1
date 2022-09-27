#import user defined functions
import time
start_time = time.time()

import numpy as np
import scipy as sp
from scipy import linalg, sparse, stats
from scipy.sparse import csr_matrix, linalg, csc_matrix
from functions import *

print("---Loading functions took %s seconds ---" % (time.time() - start_time))

# Start timer, define matrix A and b, compress matrix A to sparse matrix
n=1000
start_time = time.time()
b=np.linspace(0,10,n)**2
A = sparse.rand(n,n,density=0.1,format='csc',random_state=69)
x=np.zeros(n)
c=0.1
print("---Setting random matrix took %s seconds ---" % (time.time() - start_time))

# Banded matrix (Enable if testing banded matrix)
#k = np.diag(np.ones(n-2), k=-2) + np.diag(-3*np.ones(n-1), k=-1) + \
#    np.diag(6*np.ones(n), k=0) + \
#    np.diag(-3*np.ones(n-1), k=1) + np.diag(np.ones(n-2), k=2)
#A = csc_matrix(k)

# Preconditioner
M=precon(A,n,c)

# Linear sovler with biconjugate gradient stabilized
x = bicgstab(A,b,M)
res=sum(abs(A.dot(x)-b))
print("Final global residual:", res)
