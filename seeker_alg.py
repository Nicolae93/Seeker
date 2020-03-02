from config import Config as cf
import levy_flight as lf
import numpy as np
import threading 
import pylab
import agent
import time

def seeker_algorithm():
    # configurations
    function = cf.get_function()
    #min_domain = cf.get_min_domain()
    #max_domain = cf.get_max_domain()
    show_plots = cf.get_show_plots()
    n = cf.get_iteration()
    n_agents = cf.get_n_agents()
    scale_step = cf.get_scale_step()
    beta = cf.get_beta()
    dim = cf.get_dimension()
    global_min = cf.get_global_min()
    #creating two array for containing x and y coordinate
    #of size equals to the number of size and filled up with 0's
    if show_plots:
        x = np.zeros(n_agents)
        y = np.zeros(n_agents)
    #BOUNDS 
    bounds = cf.get_bounds()
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
    
    def reduce_domain(coordinates,reduction,bounds1):
        for i in range(dim):
            # window size
            max_domain = bounds1[i][1] 
            min_domain = bounds1[i][0]
            r_size = (np.abs(max_domain-min_domain) * reduction) * 0.5
            print('r_size',r_size)
            # modify bounds
            if coordinates[i]-r_size < bounds1[i][0]:
                bounds1[i][0] = bounds1[i][0]
            else: bounds1[i][0] = coordinates[i]-r_size   
            if coordinates[i]+r_size > bounds1[i][1]:
                bounds1[i][1] = bounds1[i][1]    
            else: bounds1[i][1] = coordinates[i]+r_size 
        print('bounds',bounds1)
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
    n_factors = number_factors(n_agents)
    n_steps = 0
    #seek algorithm
    while feval < 200000:         
        for i in range(n_agents):
            #diversification
            if count < n_factors:
                feval += n + n_steps - 1
                result = lf.levy_flight(function=function, start_coordinates=agents[i].get_best_position(),
                                        iterations=n+n_steps,bounds=bounds,show_plots=False, scale_step=scale_step, beta=beta,dim=dim,
                                        best_f=agents[i].get_best_fitness())      
            #intersification    
            else:   
                n_steps = 0
                feval += n + n_steps - 1
                result = lf.levy_flight(function=function, start_coordinates=agents[i].get_best_position(),
                                        iterations=n+n_steps,bounds=bounds,show_plots=False, scale_step=scale_step, beta=beta,dim=dim, 
                                        best_f=agents[i].get_best_fitness())
            
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
        

        
        #reduce domain
        prob = np.random.randint(10)
        if prob == 1 and count<0:  
            scale_step = cf.get_scale_step()
            print(agents[0].get_id(),agents[0].get_best_fitness(),agents[0].get_best_position(),count)
            bounds = reduce_domain(agents[0].get_best_position(),0.5,bounds)
            
            #random distribution on fitness function, refill agents in restricted domain
            for i in range(1,n_agents):
                feval += n_agents-1
                # update best point
                initial_position = []
                for j in range(dim):
                    initial_position.append(np.random.uniform(low=bounds[j][0],high=bounds[j][1]))
                agents[i].set_best_position(initial_position)
                # update best fitness
                agents[i].set_best_fitness(function(agents[i].get_best_position()))
                #plotting stuff
                if show_plots:
                    x[i],y[i] = a.get_initial_position()[0], a.get_initial_position()[1]            
            sort_agents()
        
           
        #plot
        if show_plots:  
            plots()  
        
        if success:
            break
        #update count
        count += 1   
    
    print('groupings ',n_factors)
    print('count     ',count)    
    print(agents[0].get_best_fitness(),agents[0].get_best_position())  
    print('total feval: ',feval)
    print('global min: ',global_min)
    print('x         : ',function.x)
    print()
    return [feval, success]
      

class myThread (threading.Thread):
   def __init__(self, threadID):
      threading.Thread.__init__(self)
      self.threadID = threadID
   def run(self):
      #print("Starting " + self.name)
      global s       
      # Get lock to synchronize threads
      threadLock.acquire()
      seek = seeker_algorithm()
      #start alg
      feval = seek[0]
      fitness_list.append(feval)
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
    
print('\nmean feval ',np.mean(fitness_list))   
print('percentage success '+str((s/t)*100)+'%')
print("--- %s seconds ---" % (time.time() - start_time))
