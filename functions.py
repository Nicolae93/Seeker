import numpy as np

# FUNCTIONS
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
    
def sphere(xx):
    d = len(xx)
    summa = 0
    for i in range(0,d):
        xi = xx[i]
        summa = summa + np.power(xi, 2)
    return summa

def eggholder(x):
    return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
            -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))

def rastrigin(array):
    sum = 0
    fitness = 0
    for x in array:
        sum = sum + x**2 - 10 * np.cos(2 * np.pi * x)
    fitness = 10.0 * len(array) + sum
    return fitness

def schwefel(array):
    summa = 0
    fitness = 0
    for x in array:
        summa = summa + x * np.sin(np.sqrt(np.abs(x)))
    fitness = 418.9829 * len(array) - summa
    return fitness

def michalewicz(array):#for the number of Dimension is 2
    sum = 0
    fitness = 0
    m = 10
    for (i,x) in enumerate(array, start=1):
        sum = sum + np.sin(x) * np.sin((i * (x**2) )/np.pi)**(2*m)
    fitness = -sum
    return fitness
'''
def rosen(array):
    return (1 - array[0])**2 + 100 * (array[1] - array[0]**2)**2
'''
def rosen(xx):
    d = len(xx);
    summa = 0;
    for ii in range(0,d-1):
    	xi = xx[ii]
    	xnext = xx[ii+1]
    	new = 100*(xnext-xi**2)**2 + (xi-1)**2
    	summa = summa + new
    return summa


#DOMAINS
setattr(goldstein, 'domain', [-2,2])
setattr(sphere, 'domain', [-5.12,5.12])
setattr(eggholder, 'domain', [-512,512])
setattr(rastrigin, 'domain', [-5.12,5.12])
setattr(schwefel, 'domain', [-500,500])
setattr(michalewicz, 'domain', [0,3.14])
setattr(rosen, 'domain', [-5,10])
setattr(hart3, 'domain', [0,1])

#MINIMAS
setattr(goldstein, 'min', 3.00)
setattr(sphere, 'min', 0.00)
setattr(eggholder, 'min', -959.6406)
setattr(rastrigin, 'min', 0.00)
setattr(schwefel, 'min', 0.00)
setattr(michalewicz, 'min', -1.8013)
setattr(rosen, 'min', 0.00)
setattr(hart3, 'min', -3.86277)

#X
setattr(goldstein,'x',[0,-1])
setattr(michalewicz,'x',[2.20,1.57])
setattr(sphere, 'x', [0,0])
setattr(rastrigin, 'x', [0,0])
setattr(eggholder,'x',[512,404.2319])
setattr(schwefel, 'x', [420.9687,420.9687])
setattr(rosen, 'x', [1,1])
setattr(hart3,'x',[0.114614,0.555649,0.852547])

if __name__ == '__main__':
    a = np.array([0.114614,0.555649,0.852547])
    print (hart3(a))
    