from config import Config as cf
import levy_flight as lf
import numpy as np
import pylab
import agent
import time

def seeker_algorithm():
    # configurations
    function = cf.get_function()
    min_domain = cf.get_min_domain()
    max_domain = cf.get_max_domain()
    show_plots = cf.get_show_plots()
    n = cf.get_iteration()
    n_agents = cf.get_n_agents()
    scale_step = cf.get_scale_step()
    beta = cf.get_beta()
    dim = cf.get_dimension()
    global_min = cf.get_global_min()
    #creating two array for containing x and y coordinate
    #of size equals to the number of size and filled up with 0's
    x = np.zeros(n_agents)
    y = np.zeros(n_agents)
    #BOUNDS 
    bounds = [min_domain, max_domain]
    #initial evaluation
    feval = n_agents
    agents = []                 
    
    def print_info():
        for a in agents:
            print(a.get_id(),a.get_best_fitness(),a.get_best_position())
            
    def sort_agents():
        agents.sort(key=lambda x: x.get_best_fitness())
    
    def number_factors(x):
        n = 0
        for i in range(1, x + 1):
            if x % i == 0:
                n += 1
        return n-1        
    
    def plots():
        # plotting stuff:
        pylab.title("Seeker alg ($n = " + str(n) + "$ steps)")
        pylab.plot(x, y,'o',ms=4)
        #pylab.savefig("Pure_Random_Search"+str(n)+".png",bbox_inches="tight",dpi=600)
        pylab.show() 
    
    #random distribution on fitness function
    for i in range(n_agents):
        a = agent.Agent(function=function,bounds=bounds,my_id=i,dim=dim)
        agents.append(a)     
        #plotting stuff
        if show_plots:
            x[i],y[i] = a.get_initial_position()[0], a.get_initial_position()[1]            
    sort_agents()
    
    #print_info()
    
    #plot
    if show_plots: 
        plots()
   
    count = 0    
    success = False
    n_factors = number_factors(n_agents)
    #seek algorithm
    while feval < 200000:         
        for i in range(n_agents):
            #diversification
            if count < n_factors:
                feval += n+10
                result = lf.levy_flight(function=function, start_coordinates=agents[i].get_best_position(),
                           iterations=n+10,bounds=bounds,show_plots=False, scale_step=scale_step, beta=beta,dim=dim)
            #intersification    
            else:   
                feval += n
                result = lf.levy_flight(function=function, start_coordinates=agents[i].get_best_position(),
                               iterations=n,bounds=bounds,show_plots=False, scale_step=scale_step, beta=beta,dim=dim)
            
            if result.fun < agents[i].get_best_fitness(): # if there are some improvements in the levy walk, update
                # update best fitness
                agents[i].set_best_fitness(result.fun)
                # update best point
                agents[i].set_best_position(result.x)
            #plotting stuff 
            if show_plots:    
                x[i],y[i] = agents[i].get_best_position()[0], agents[i].get_best_position()[1]   
            #check if global min is reached
            if(np.abs(result.fun-global_min) <= 0.00001):
                print('SUCCESS')
                success = True
                break
        sort_agents() 
        
        #change position for help
        if count < n_factors:
            #change position for help
            for i in range(0,int(len(agents)/2)):
                agents[-(i+1)].set_best_position(agents[i].get_best_position()) 
                agents[-(i+1)].set_best_fitness(agents[i].get_best_fitness())    
            sort_agents()
        else:
            for i in range(1, len(agents)):
                agents[(i)].set_best_position(agents[0].get_best_position()) 
                agents[(i)].set_best_fitness(agents[0].get_best_fitness())   

        #decrease scale_step by 10%
        scale_step = scale_step - scale_step*0.1
        
        #plot
        if show_plots: 
            plots()  
        
        if success:
            break   
        count += 1
        #decrease scale_step by 10%
        scale_step = scale_step - scale_step*0.1
        
        #plot
        if show_plots: 
            plots()  
        
        if success:
            break   
        count += 1
        
           
    print(agents[0].get_best_fitness(),agents[0].get_best_position())  
    print('total feval: ',feval)
    print('global min: ',global_min)
    print('x         : ',function.x)
    
    return [feval, success]

start_time = time.time()
r = 30
c=0
fitness_list = []
for i in range(r):
    seek = seeker_algorithm()
    feval = seek[0]
    fitness_list.append(feval)
    if seek[1]:
        c += 1
        
print('\nmean feval ',np.mean(fitness_list))   
print('percentage success '+str((c/r)*100)+'%')
print("--- %s seconds ---" % (time.time() - start_time))