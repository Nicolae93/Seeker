import functions as fn

class Config:
    __Function = fn.hart3
    __Dimension = 3 # The number of dimension
    __GlobalMin = __Function.min
    __MinDomain = __Function.domain[0] # variable lower limit
    __MaxDomain = __Function.domain[1] # variable upper limit
    #__Bounds = np.array(__Function.domain) * __Dimension
    __Bounds = __Function.domain
    __Beta = 1.4 # parameter for Levy flight
    __Show_Plots = False
    __Iteration = 2
    __Agents = 1
    __ScaleStep = 1
    
    @classmethod
    def get_bounds(cls):
        return cls.__Bounds
    
    @classmethod
    def get_n_agents(cls):
        return cls.__Agents
    
    @classmethod
    def get_show_plots(cls):
        return cls.__Show_Plots
    
    @classmethod
    def get_scale_step(cls):
        return cls.__ScaleStep
    
    @classmethod
    def get_show_multiple_run_plots(cls):
        return cls.__ShowMultipleRunPlots
    
    @classmethod
    def get_show_single_run_plots(cls):
        return cls.__ShowSingleRunPlots
    
    @classmethod
    def get_global_min(cls):
        return cls.__GlobalMin
        
    @classmethod
    def get_function(cls):
        return cls.__Function
    
    @classmethod
    def set_function(cls,_function):
        cls.__Function = _function
    
    @classmethod
    def get_iteration(cls):
        return cls.__Iteration

    @classmethod
    def get_dimension(cls):
        return cls.__Dimension

    @classmethod
    def get_max_domain(cls):
        return cls.__MaxDomain

    @classmethod
    def set_max_domain(cls, _max_domain):
        cls.__MaxDomain = _max_domain

    @classmethod
    def get_min_domain(cls):
        return cls.__MinDomain

    @classmethod
    def set_min_domain(cls, _min_domain):
        cls.__MinDomain = _min_domain

    @classmethod
    def get_beta(cls):
        return cls.__Beta

    @classmethod
    def set_beta(cls, _beta):
        cls.__Beta = _beta
