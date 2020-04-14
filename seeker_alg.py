from config import Config as cf
import levy_flight as lf
import numpy as np
import threading 
import pylab
import agent
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def seeker_algorithm():
    # configurations
    function = cf.get_function()
    show_plots = cf.get_show_plots()
    n = cf.get_iteration()
    n_agents = cf.get_n_agents()
    scale_step = cf.get_scale_step()
    beta = cf.get_beta()
    dim = cf.get_dimension()
    global_min = cf.get_global_min()
    bounds = cf.get_bounds()
    plot_3D_function = cf.get_plot_3d_func()
    
    #creating two array for containing x and y coordinate
    #of size equals to the number of size and filled up with 0's
    if show_plots:
        x = np.zeros(n_agents)
        y = np.zeros(n_agents)
    #PLOT 3D function     
    if plot_3D_function:
        x = np.arange(bounds[0][0], bounds[0][1]+1)
        y = np.arange(bounds[0][0], bounds[0][1]+1)
        xgrid, ygrid = np.meshgrid(x, y)
        xy = np.stack([xgrid, ygrid])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(45, -45)
        ax.plot_surface(xgrid, ygrid, function(xy), cmap='terrain')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('goldstein-price(x, y)')
        #plt.savefig('3Dfunction.png',dpi=600)
        plt.show()
        
    #initial evaluation
    feval = n_agents
    agents = []                 
    
    def print_info():
        for a in agents:
            print(a.get_id(),a.get_best_fitness(),a.get_best_position())
        print()    
        
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
    
    def reduce_domain(coordinates,reduction,current_bounds):
        bounds1 = current_bounds.copy()
        for i in range(dim):
            # window size
            max_domain = bounds1[i][1] 
            min_domain = bounds1[i][0]
            # check if it is too small
            if abs(max_domain-min_domain) < 0.1:
                print('Too small')
                #return current_bounds
                return cf.get_bounds()
            r_size = (np.abs(max_domain-min_domain) * reduction) / 2
            #print('r_size',r_size)
            # modify bounds
            if coordinates[i]-r_size < bounds1[i][0]:
                bounds1[i][0] = bounds1[i][0]
            else: bounds1[i][0] = coordinates[i]-r_size   
            if coordinates[i]+r_size > bounds1[i][1]:
                bounds1[i][1] = bounds1[i][1]    
            else: bounds1[i][1] = coordinates[i]+r_size 
        #print('bounds',bounds1)
        return bounds1
    
    #random distribution on fitness function
    for i in range(n_agents):
        a = agent.Agent(function=function,bounds=bounds,my_id=i,dim=dim)
        agents.append(a)     
        #plotting stuff
        if show_plots:
            x[i],y[i] = a.get_initial_position()[0], a.get_initial_position()[1]            
    sort_agents()
    
    #print_info()
    #print(agents[0].get_id(),agents[0].get_best_fitness(),agents[0].get_best_position())
    
    #plot
    if show_plots: 
        plots()
   
    count = 0    
    success = False
    #n_factors = number_factors(n_agents)
    n_steps = 0 # per controllare gli steps nel caso volessi fare più passi levy in futuro
    
    prob_number = 70
    #seek algorithm
    while feval < 2000000:         
        for i in range(n_agents):
            #diversification            
            result = lf.levy_flight(function=function, start_coordinates=agents[i].get_best_position(),
                                        iterations=n+n_steps,bounds=bounds,show_plots=False, scale_step=scale_step, beta=beta,dim=dim,
                                        best_f=agents[0].get_best_fitness(), fevalu=feval) 
            #if result.feval!=0:
                #print(result.feval, agents[0].get_best_fitness())
                
            feval += n + n_steps - 1 + result.feval
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
        
        #reduce n steps
        n_steps = int(n_steps - n_steps * 0.5)
        
        #decrease scale_step by 10%
        #print(scale_step)
        if scale_step > 0.0001:
            scale_step = scale_step - scale_step * 0.1
            
        
        #change position for help
        for i in range(1, len(agents)):
                agents[(i)].set_best_position(agents[0].get_best_position()) 
                agents[(i)].set_best_fitness(agents[0].get_best_fitness()) 
        
        '''        
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
        '''        
        
        # Compute probability to allow diversification        
        prob = np.random.randint(prob_number)
            
        if prob == 1:  
            #prob_number += 1
            scale_step = cf.get_scale_step()
            
            '''
            #BOUND REDUCTION
            prob = np.random.randint(10)
            if prob == 1:  
                print('reduction!')
                reduct = 0.5
                #reduct = 1-np.finfo(np.float64).eps
                #print(reduct)
                bounds = reduce_domain(agents[0].get_best_position(),reduct,bounds)
            else: bounds = cf.get_bounds()
            '''
            if n_agents > 1:
                #random distribution on fitness function, refill agents in restricted domain
                for i in range(1,n_agents):
                    # update best point
                    initial_position = []
                    for j in range(dim):
                        initial_position.append(np.random.uniform(low=bounds[j][0],high=bounds[j][1]))
                    feval += n_agents-1                    
                    agents[i].set_best_position(initial_position)
                    agents[i].set_best_fitness(function(agents[i].get_best_position()))
                sort_agents()
                
            else:
                initial_position = []
                for j in range(dim):
                    initial_position.append(np.random.uniform(low=bounds[j][0],high=bounds[j][1]))
                feval += 1
                current_fun = function(initial_position)
                if current_fun < agents[0].get_best_fitness():
                    agents[0].set_best_position(initial_position)
                    agents[0].set_best_fitness(current_fun)
            
        #plot
        if show_plots:  
            plots()  
        
        if success:
            break
        #update count
        count += 1   
    
    print(agents[0].get_best_fitness())#,agents[0].get_best_position())  
    print('total FEVAL: ',feval)
    print('global min: ',global_min)
    #print('x         : ',function.x)
    print()
    return [feval, success, agents[0].get_best_fitness()]
      

class myThread (threading.Thread):
   def __init__(self, threadID):
      threading.Thread.__init__(self)
      self.threadID = threadID
   def run(self):
      #print("Starting " + self.name)
      global s       
      # Get lock to synchronize threads
      seek = seeker_algorithm()
      threadLock.acquire()
      #start alg
      feval = seek[0]
      fitness_list.append(feval)
      record = seek[2]
      record_list.append(record)
      if seek[1]:
          s += 1
      #end alg    
          
      # Free lock to release next thread
      threadLock.release()    
      #print()    
        
start_time = time.time()      
threadLock = threading.Lock()
threads = []
s = 0 
fitness_list = []
record_list = []

t = 1
for i in range(t):
    # Create new threads
    thread = myThread(i)
    # Start new Threads
    thread.start()
    # Add threads to thread list
    threads.append(thread)

# Wait for all threads to complete
for th in threads:
    th.join()
#print("Exiting Main Thread\n")      
    
print('\nmean feval                ',np.mean(fitness_list))   
print('standard deviation feval  ',int(np.std(fitness_list)))
print('percentage success '+str((s/t)*100)+'%')
print('mean record               ', np.mean(record_list))
print('standard deviation record ',int(np.std(record_list)))
print("--- %s seconds ---" % (time.time() - start_time))
