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
                bounds=[0, 1],scale_step = 0.1, show_plots=False, beta=1.5, best_f = 3, fevalu=3):
    feval = 0
    n = iterations;
    coordinates = []
    coordinates_inverse = []
    for i in range(dim):
        coordinates.append(np.zeros(n))
        coordinates_inverse.append(np.zeros(n))
        #set start coordinates
        coordinates[i][0] = start_coordinates[i]
        coordinates_inverse[i][0] = start_coordinates[i]

    #call levy steps
    d = levy_flight_steps(beta=beta, n=iterations, m=dim)
    #for i in range(len(d)):
        #d[i] = -d[i]
    #print(d)
    #print(d*scale_step)
    # set minimum
    minimum = best_f
    best_point = start_coordinates
    
    f_points = [minimum]
    val_i = 0
    flag = True
    try_inverse = True
    
    def fix_bounds():    
        temp_scale_step = scale_step
        while not all(coordinates_bound_check): 
            #decrease temp_scale_step by 10%
            temp_scale_step = temp_scale_step - temp_scale_step * 0.1
            # if the step is too long try to make a smaller step
            for j in range(dim):
                if not coordinates_bound_check[j]:
                    coordinates[j][i] = coordinates[j][i-1]+d[val_i+j]*temp_scale_step
                    coordinates_bound_check[j] = bounds[j][0]< coordinates[j][i] <bounds[j][1]  
        return coordinates            
        
    def fix_bounds_inverse():
        temp_scale_step = scale_step
        while not all(coordinates_bound_check_inverse): 
            #decrease temp_scale_step by 10%
            temp_scale_step = temp_scale_step - temp_scale_step * 0.1
            # if the step is too long try to make a smaller step
            for j in range(dim):
                if not coordinates_bound_check_inverse[j]:
                    coordinates_inverse[j][i] = coordinates_inverse[j][i-1]+d[val_i+j]*temp_scale_step
                    coordinates_bound_check_inverse[j] = bounds[j][0]< coordinates_inverse[j][i] <bounds[j][1]  
        return coordinates_inverse     
    
    while flag:
        for i in range(1,n): # use those steps 
            coordinates_bound_check = []
            coordinates_bound_check_inverse = []
            curr_point = []
            curr_point_inverse = []
            # make a step without careing if it is out of bounds
            for j in range(dim):
                next_point = coordinates[j][i-1]+d[val_i+j]*scale_step
                coordinates[j][i] = next_point
                coordinates_bound_check.append(bounds[j][0]<next_point<bounds[j][1])
                if i == 1:
                    next_point_inverse = coordinates_inverse[j][i-1]-d[val_i+j]*scale_step
                    coordinates_inverse[j][i] = next_point_inverse
                    coordinates_bound_check_inverse.append(bounds[j][0]<next_point_inverse<bounds[j][1])
                
            # take care of out of bound coordinates
            fix_bounds()  
            fix_bounds_inverse()
            
            for j in range(dim):
                curr_point.append(coordinates[j][i])
                curr_point_inverse.append(coordinates_inverse[j][i])
            
            # get current function evaluation
            f_curr_point = function(curr_point)
            feval += 1
            
            #print (minimum,f_curr_point,f_curr_point_inverse)
            #print(curr_point,curr_point_inverse)
            
            #check if current point is better than current minimum 
            if f_curr_point < minimum:
                print('bingo',f_curr_point,feval,fevalu,scale_step)
                #print(f_curr_point)
                f_points.append(f_curr_point)
                minimum = f_curr_point
                best_point = curr_point  
                scale_step = scale_step + scale_step*1
                #print(scale_step)
                #print('bingo')
                try_inverse = False
            elif try_inverse:
                # try to see if in the other direction is better
                f_curr_point_inverse = function(curr_point_inverse)
                feval += 1
                
                if f_curr_point_inverse < minimum:
                    print('bongo',f_curr_point_inverse, feval,fevalu, scale_step)
                    #print(f_curr_point_inverse)
                    f_points.append(f_curr_point_inverse)
                    minimum = f_curr_point_inverse
                    best_point = curr_point_inverse  
                    scale_step = scale_step + scale_step*1
                    for j in range(dim):
                        d[val_i+j] = -d[val_i+j]
                        coordinates[j][i] = coordinates_inverse[j][i]
                    #print('bongo')
                try_inverse = False
            else: 
                flag = False 
                val_i = val_i + 2    
            '''
            if f_curr_point < minimum:
                #print(f_curr_point,feval,fevalu)
                #print(f_curr_point)
                f_points.append(f_curr_point)
                minimum = f_curr_point
                best_point = curr_point  
                scale_step = scale_step + scale_step*0.5
                #print('bingo')
                # se non vuoi migliorare il risultato aggiunto metti flag=false
                #flag = False 
            else: 
                flag = False 
                val_i = val_i + 2        
            '''  
    f_points.append(f_points[-1])
    #create an optResult object
    result = optimize.OptimizeResult(x=best_point,fun=minimum,f_points=f_points, feval=feval)    
    #print('true iterations: ', count)
    if show_plots:
        # plotting stuff:
        pylab.title("Levi-Flight ($n = " + str(n) + "$ steps)")
        pylab.plot(coordinates[0],coordinates[1],'-',lw=0.5)
        #pylab.savefig("Levi-Flight"+str(n)+".png",bbox_inches="tight",dpi=600)
        pylab.show()    
    
    return result
'''
levy_flight(function=fn.goldstein, start_coordinates=[0,0], iterations=10, bounds=[[-2, 2],[-2, 2]], 
            show_plots=True, dim=2, best_f = 600, scale_step = .1, beta=1.5)
'''
'''
levy_flight(function=fn.eggholder, start_coordinates=[0,0], iterations=10, bounds=[[-512,512],[-512,512]], 
            show_plots=True, dim=2, best_f = -25.46033718528632, scale_step = 10, beta=1.5)
'''

