import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum, power
# FUNCTIONS
'''
alpha = [1.0, 1.2, 3.0, 3.2]';
A = [10, 3, 17, 3.5, 1.7, 8;
     0.05, 10, 17, 0.1, 8, 14;
     3, 3.5, 1.7, 10, 17, 8;
     17, 8, 0.05, 10, 0.1, 14];
P = 10^(-4) * [1312, 1696, 5569, 124, 8283, 5886;
               2329, 4135, 8307, 3736, 1004, 9991;
               2348, 1451, 3522, 2883, 3047, 6650;
               4047, 8828, 8732, 5743, 1091, 381];

outer = 0;
for ii = 1:4
	inner = 0;
	for jj = 1:6
		xj = xx(jj);
		Aij = A(ii, jj);
		Pij = P(ii, jj);
		inner = inner + Aij*(xj-Pij)^2;
	end
	new = alpha(ii) * exp(-inner);
	outer = outer + new;
end

y = -(2.58 + outer) / 1.94;
'''
def hart6(xx):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],[0.05, 10, 17, 0.1, 8, 14],[3, 3.5, 1.7, 10, 17, 8],[17, 8, 0.05, 10, 0.1, 14]])
    P = np.multiply(10**(-4),np.array([[1312, 1696, 5569, 124, 8283, 5886],[2329, 4135, 8307, 3736, 1004, 9991],[2348, 1451, 3522, 2883, 3047, 6650],[4047, 8828, 8732, 5743, 1091, 381]]))
    outer = 0
    for ii in range(0,4):
        inner = 0
        for jj in range(0,6):
            xj = xx[jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner = inner + Aij*(xj-Pij)**2
        new = alpha[ii] * np.exp(-inner)
        outer += new
    return -outer
 
def hart3(xx):
    alpha = np.array([1.0, 1.2, 3.0, 3.2]).T
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
        
def goldstein(xy):
    '''
    Goldstein-Price Function
    global minimum: f(0, -1) = 3
    bounds: -2 <= x, y <= 2
    '''
    x, y = xy[0], xy[1]
    return ((1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) *
            (30 + (2*x-3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)))
    
def sphere(x):
    '''
    Sphere Function
    global minimum at x=0 where f(x)=0
    bounds: none
    '''
    return sum([item * item for item in x])

def eggholder(xy):
    '''
    Eggholder Function
    global minimum: f(x=512, y=404.2319) = -959.6407
    bounds: -512 <= x, y <= 512
    '''
    x, y = xy[0], xy[1]
    return (-(y+47)*sin(sqrt(abs((x/2.0) + (y+47)))) -
            x*sin(sqrt(abs(x-(y+47)))))

def rastrigin(x, safe_mode=False):
    '''
    Rastrigin Function
    wikipedia: https://en.wikipedia.org/wiki/Rastrigin_function
    global minimum at x=0, where f(x)=0
    bounds: -5.12 <= x_i <= 5.12
    '''
    if safe_mode:
        for item in x: assert x<=5.12 and x>=-5.12, 'input exceeds bounds of [-5.12, 5.12]'
    return len(x)*10.0 +  sum([item*item - 10.0*cos(2.0*pi*item) for item in x])

def schwefel(x):
    '''
    Schwefel Function
    reference: https://www.sfu.ca/~ssurjano/schwef.html
    global minimum: f(x) = 0 when x = [420.9687,...,420.9687]
    bounds: -500 <= x_i <= 500
    '''
    return 418.9829*len(x) - sum([item * sin(sqrt(abs(item))) for item in x])


def michalewicz( x ):  # mich.m
    michalewicz_m = 10  # orig 10: ^20 => underflow
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    return - sum( sin(x) * sin( j * x**2 / pi ) ** (2 * michalewicz_m) )

def rosen(x):
    '''
    Rosenbrock Function
    wikipedia: https://en.wikipedia.org/wiki/Rosenbrock_function
    global minimum:
        f(x)=0 where x=[1,...,1]
    bounds:
        -inf <= x_i <= +inf
        1 <= i <= n
    '''

    total = 0
    for i in range(len(x)-1):
        total += 100*(x[i+1] - x[i]*x[i])**2 + (1-x[i])**2
    return total

def zakharov( x ):  # zakh.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    s2 = sum( j * x ) / 2
    return sum( x**2 ) + s2**2 + s2**4

#DOMAINS
setattr(goldstein, 'domain', [[-2,2],[-2,2]])
setattr(eggholder, 'domain', [[-512,512],[-512,512]])
setattr(michalewicz, 'domain', [[0,pi],[0,pi]])
setattr(schwefel, 'domain', [[-500,500],[-500,500]])
setattr(hart3, 'domain', [[0,1],[0,1],[0,1]])
setattr(hart6, 'domain', [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])
setattr(sphere, 'domain', [[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12]])
setattr(rastrigin, 'domain', [[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12],[-5.12,5.12]])
setattr(rosen, 'domain', [[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10]])
#setattr(rosen, 'domain', [[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10]])
setattr(zakharov, 'domain', [[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10],[-5,10]])


#MINIMAS
setattr(goldstein, 'min', 3)
setattr(eggholder, 'min', -959.6406627106155)
setattr(michalewicz, 'min', -1.8012982949924439)
setattr(schwefel, 'min', 0.00002545567497236334)
setattr(hart3, 'min', -3.8627797869493365)
setattr(hart6, 'min', -3.322368011391339)
setattr(sphere, 'min', 0.0000000000000)
setattr(rastrigin, 'min', 0)
setattr(rosen, 'min', 0)


#X
setattr(goldstein,'x',[0,-1])
setattr(eggholder,'x',[512,404.2319])
setattr(michalewicz,'x',[2.20319,1.57049])
setattr(schwefel, 'x', [420.9687,420.9687])
setattr(hart3,'x',[0.114614,0.555649,0.852547])
setattr(hart6,'x',[0.20169,0.150011,0.476874,0.275332,0.311652,0.6573])
setattr(sphere, 'x', [0,0])
setattr(rastrigin, 'x', [0,0])
setattr(rosen, 'x', [1,1])

if __name__ == '__main__':
    a = np.array( [0.20169,0.150011,0.476874,0.275332,0.311652,0.6573] )
    print (hart6(a))