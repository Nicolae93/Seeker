import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
# FUNCTIONS
#def hart6(xx):
    
def hart3(xx):
    alpha = np.array([1.0, 1.2, 3.0, 3.2]).T;
    A = np.array([[3.0, 10, 30],[0.1, 10, 35],[3.0, 10, 30],[0.1, 10, 35]])
    P = np.multiply(10**(-4),np.array([[3689, 1170, 2673],[4699, 4387, 7470],[1091, 8732, 5547],[381, 5743, 8828]]))
    outer = 0
    for ii in range(0,4):
        inner = 0
        for jj in range(0,3):
            xj = xx[jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner = inner + Aij*(xj-Pij)**2
        new = alpha[ii] * np.exp(-inner)
        outer += new
    return -outer
        
def goldstein(xx):
    x1 = xx[0]
    x2 = xx[1]
    fact1a = (x1 + x2 + 1)**2
    fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
    fact1 = 1 + fact1a*fact1b
    fact2a = (2*x1 - 3*x2)**2
    fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
    fact2 = 30 + fact2a*fact2b
    return fact1*fact2
    
def sphere( x ):
    x = np.asarray_chkfinite(x)
    return sum( x**2 )

def eggholder(x):
    return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
            -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))

def rastrigin( x ):  # rast.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 10*n + sum( x**2 - 10 * cos( 2 * pi * x ))

def schwefel( x ):  # schw.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 418.9829*n - sum( x * sin( sqrt( abs( x ))))


def michalewicz( x ):  # mich.m
    michalewicz_m = 10  # orig 10: ^20 => underflow
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    return - sum( sin(x) * sin( j * x**2 / pi ) ** (2 * michalewicz_m) )

def rosen( x ):  
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    return (sum( (1 - x0) **2 )
        + 100 * sum( (x1 - x0**2) **2 ))


#DOMAINS
setattr(goldstein, 'domain', [[-2,2],[-2,2]])
setattr(eggholder, 'domain', [[-512,512],[-512,512]])
setattr(michalewicz, 'domain', [[0,pi],[0,pi]])
setattr(schwefel, 'domain', [[-500,500],[-500,500]])
setattr(hart3, 'domain', [[0,1],[0,1],[0,1]])
setattr(sphere, 'domain', [[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12]])
setattr(rastrigin, 'domain', [[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12]])
setattr(rosen, 'domain', [[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10]])

#MINIMAS
setattr(goldstein, 'min', 3.000000000000000)
setattr(eggholder, 'min', -959.6406627106155)
setattr(michalewicz, 'min', -1.8012982949924439)
setattr(schwefel, 'min', 0.00002545567497236334)
setattr(hart3, 'min', -3.8627797869493365)
setattr(sphere, 'min', 0.0000000000000)
setattr(rastrigin, 'min', 0.000000000000)
setattr(rosen, 'min', 0.0000000000000)

#X
setattr(goldstein,'x',[0,-1])
setattr(eggholder,'x',[512,404.2319])
setattr(michalewicz,'x',[2.20319,1.57049])
setattr(schwefel, 'x', [420.9687,420.9687])
setattr(hart3,'x',[0.114614,0.555649,0.852547])
setattr(sphere, 'x', [0,0])
setattr(rastrigin, 'x', [0,0])
setattr(rosen, 'x', [1,1])

if __name__ == '__main__':
    a = np.array( [512,404.2319] )
    print (eggholder(a))
    