from config import Config as cf
import levy_flight as lf
import numpy as np
import pylab
import agent

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
    #creating two array for containing x and y coordinate
    #of size equals to the number of size and filled up with 0's
    x = np.zeros(n_agents)
    y = np.zeros(n_agents)
    #BOUNDS 
    bounds = [(min_domain, max_domain), (min_domain, max_domain)]
    #initial evaluation
    feval = n_agents
    agents = [] 
    best_fitness = []
    best_points = []   
    #random distribution on fitness function
    for i in range(n_agents):
        a = agent.Agent(function=function,bounds=bounds,my_id=i)
        agents.append(a)     
        best_fitness.append(a.get_initial_fitness())
        best_points.append(a.get_initial_position())
        #plotting stuff
        x[i],y[i] = a.get_initial_position()[0], a.get_initial_position()[1]              
    
    def print_info():
        for a in agents:
            print(a.get_id(),a.get_best_fitness(),a.get_best_position())
            
    def sort_agents():
        agents.sort(key=lambda x: x.get_best_fitness())
        for i in range(n_agents):
            best_fitness[i] = agents[i].get_best_fitness()
            best_points[i] = agents[i].get_best_position()
            
    #print_info()   
    print('----start-------------------------------------------------')
    sort_agents()
    #print_info() 
    
    if show_plots: 
        # plotting stuff:
        pylab.title("Seeker alg ($n = " + str(n) + "$ steps)")
        pylab.plot(x, y,'o',ms=4)
        #pylab.savefig("Pure_Random_Search"+str(n)+".png",bbox_inches="tight",dpi=600)
        pylab.show()     
    print('----agent reorder-----------------------------------------')
    
    #make 10 levy-flight steps
    for i in range(n_agents):
        feval += n
        result = lf.levy_flight(function=function, start_coordinates=agents[i].get_best_position(),
                       iterations=n,bounds=bounds,show_plots=False, scale_step=scale_step,beta=beta)
        if result.fun < best_fitness[i]: # if there are some improvements in the levy walk, update
            # update best fitness
            agents[i].set_best_fitness(result.fun)
            best_fitness[i] = result.fun
            # update best point
            agents[i].set_best_position(result.x)
            best_points[i] = result.x  
        #plotting stuff    
        x[i],y[i] = agents[i].get_best_position()[0], agents[i].get_best_position()[1]    
    
    sort_agents()
    #print_info() 
    print('----first levy moves--------------------------------------')
    
    if show_plots: 
        # plotting stuff:
        pylab.title("Seeker alg 1 ($n = " + str(n) + "$ steps)")
        pylab.plot(x, y,'o',ms=4)
        #pylab.savefig("Pure_Random_Search"+str(n)+".png",bbox_inches="tight",dpi=600)
        pylab.show()  
       
    def compute_groupings(n):
        count = 0
        while n != 1:
            n = n/2
            count += 1
        return count
        
    global_min = cf.get_global_min()
    #groupings = compute_groupings(n_agents)   
    #groupings += 20
    success = False
    
    #for e in range(200000):    
    while feval < 200000:
        #change position for help
        for i in range(0,int(len(agents)/2)):
            agents[-(i+1)].set_best_position(agents[i].get_best_position()) 
            agents[-(i+1)].set_best_fitness(agents[i].get_best_fitness())
        
        sort_agents() 
        #print_info()
        #print('----split '+str(e)+' ----------------------------------------------')
        
        for i in range(n_agents):
            feval += n
            result = lf.levy_flight(function=function, start_coordinates=agents[i].get_best_position(),
                           iterations=n,bounds=bounds,show_plots=False, scale_step=scale_step, beta=beta)
            
            if result.fun < best_fitness[i]: # if there are some improvements in the levy walk, update
                # update best fitness
                agents[i].set_best_fitness(result.fun)
                best_fitness[i] = result.fun
                # update best point
                agents[i].set_best_position(result.x)
                best_points[i] = result.x  
            x[i],y[i] = agents[i].get_best_position()[0], agents[i].get_best_position()[1]    
            #check if global min is reached
            if(np.abs(result.fun-global_min) <= 0.00001):
                print('SUCCESS')
                success = True
                break
        
        
        sort_agents() 
        #print_info()
        #print('----split '+str(e)+' ----------------------------------------------')
        
        #decrease scale_step by 10%
        scale_step = scale_step - scale_step*0.1
        #print('scale_step ',scale_step)
        
        if show_plots: 
            # plotting stuff:
            pylab.title("Seeker split ($n = " + str(n) + "$ steps)")
            pylab.plot(x, y,'o',ms=4)
            #pylab.savefig("Pure_Random_Search"+str(n)+".png",bbox_inches="tight",dpi=600)
            pylab.show()  
            
        if success:
            break    
            
    print(best_fitness[0],best_points[0])  
    print('total feval: ',feval)
    print('global min: ',global_min)
    print('x         : ',function.x)
    
    return 1

seeker_algorithm()