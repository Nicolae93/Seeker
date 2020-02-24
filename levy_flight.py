import numpy as np
import math
import pylab
import functions as fn
from scipy import optimize

# Input parameters
# n     -> Number of steps 
# m     -> Number of Dimensions 
# beta  -> Power law index  % Note: 1 < beta < 2

def levy_flight_steps(beta, n=100000, m=2):
    
    num = math.gamma(1+beta)*np.sin(np.pi*beta/2); # used for Numerator 
    
    den = math.gamma((1+beta)/2)*beta*2**((beta-1)/2); # used for Denominator

    sigma_u = (num/den)**(1/beta); # Standard deviation
    
    u = np.random.normal(0, sigma_u, n*m)

    v = np.random.normal(0, 1, n*m)
    
    z = u / np.power(np.fabs(v), 1 / beta)

    return z

def levy_flight(function=fn.hart3, dim=3, start_coordinates=[0.5,0.5,0.5], iterations=3, 
                bounds=[0, 1],scale_step = 0.1, show_plots=False, beta=1.5):
    n = iterations;
    coordinates = []
    for i in range(dim):
        coordinates.append(np.zeros(n))
        #set start coordinates
        coordinates[i][0] = start_coordinates[i]

    #call levy steps
    d = levy_flight_steps(beta=beta, n=iterations, m=dim) 
    # set minimum
    minimum = function(start_coordinates)
    best_point = start_coordinates
    
    f_points = [minimum]
    val_i = 0
    
    for i in range(1,n): # use those steps 
        coordinates_bound_check = []
        temp_scale_step = scale_step
        # set next step without careing if it is out of bounds
        for j in range(dim):
            coordinates[j][i] = coordinates[j][i-1]+d[val_i+j]*scale_step
            coordinates_bound_check.append(bounds[0]<coordinates[j][i-1]+d[val_i+j]*scale_step<bounds[1])
        # take care of the out of bound coordinates    
        while not all(coordinates_bound_check): 
            #decrease temp_scale_step by 10%
            temp_scale_step = temp_scale_step - temp_scale_step*0.1
            # if the step is too long try to make a smaller step
            for j in range(dim):
                if not coordinates_bound_check[j]:
                    coordinates[j][i] = coordinates[j][i-1]+d[val_i+j]*temp_scale_step
                    coordinates_bound_check[j] = bounds[0]< coordinates[j][i] <bounds[1]       
        val_i = val_i + 2
        
        #check if current point is better than current minimum 
        curr_point = []
        for j in range(dim):
            curr_point.append(coordinates[j][i])
        f_curr_point = function(curr_point)
        if  f_curr_point <= minimum:
            f_points.append(f_curr_point)
            minimum = f_curr_point
            best_point = curr_point    
            
    f_points.append(f_points[-1])
    #create an optResult object
    result = optimize.OptimizeResult(x=best_point,fun=minimum,f_points=f_points)    
    #print('true iterations: ', count)
    if show_plots:
        # plotting stuff:
        pylab.title("Levi-Flight ($n = " + str(n) + "$ steps)")
        pylab.plot(coordinates[0],coordinates[1],'-',lw=0.5)
        #pylab.savefig("Levi-Flight"+str(n)+".png",bbox_inches="tight",dpi=600)
        pylab.show()    
    return result    

#levy_flight(function=fn.goldstein, start_coordinates=[0,0], iterations=3, bounds=[-2, 2], show_plots=True, dim=2)