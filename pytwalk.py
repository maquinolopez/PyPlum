#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
########################################################
This is the python implementation of the t-walk ### 
By J Andres Christen.                             ###
A generic self adjusting MCMC                   ###
see:  http://www.cimat.mx/~jac/twalk/           ###

###  New version March 2023 ###
New plotting, saving and loading methods and also
### new derived classes for Bayesian inference
 pyPstwalk, Ind1dsampl and BUQ.
 
###  see new twalktutorial2.py ###
########################################################

"""

import signal

from time import time, localtime, strftime

from numpy import array, ceil, zeros, linspace, loadtxt, savetxt, arange, mod
from numpy import exp, where, cov, ones, sqrt, log, floor, pi, cumsum, append
from numpy import sum as np_sum
from scipy.stats import uniform, norm
from tqdm import tqdm 
try:
    from matplotlib.pylab import subplots
except:
    print("pytwalk: matplotlib.pylab module not found, needed for plot methods.")

try:
    from corner import corner
except:
    print("pytwalk: corner module not found, needed for method PlotCorner.\n Install it with: pip install corner.")




############################################################################################
#### Auxiliary functions to calculate Integrated autocorrelation times of a time series 


def AutoCov( Ser, c, la, T=0):
    """
    ####  Calculates an autocovariance 2x2 matrix at lag l in column c of matrix Ser with T rows
    ####  The variances of each series are in the diagonal and the (auto)covariance in the off diag.
    """
    if (T == 0):
        T = Ser.shape[0]  ### Number of rows in the matrix (sample size)

    return cov( Ser[0:(T-1-la), c], Ser[la:(T-1), c], bias=1)
    
    
    
from numpy import shape, mat, matrix
def AutoCorr( Ser, cols=0, la=1):
    """
    #### Calculates the autocorrelation from lag 0 to lag la of columns cols (list)
    #### for matrix Ser
    """
    T = Ser.shape[0]  ### Number of rows in the matrix (sample size)

    ncols = shape(mat(cols))[1] ## Number of columns to analyse (parameters)

    #if ncols == 1:
    #    cols = [cols]
        
    ### Matrix to hold output
    Out = matrix(ones((la+1)*ncols)).reshape( la+1, ncols)
        
    for c in range(ncols):
        for l in range( 1, la+1):  
            Co = AutoCov( Ser, cols[c], l, T) 
            Out[l,c] = Co[0,1]/(sqrt(Co[0,0]*Co[1,1]))
    
    return Out
    

### Makes an upper band matrix of ones, to add the autocorrelation matrix
### gamma = auto[2*m+1,c]+auto[2*m+2,c] etc. 
### MakeSumMat(lag) * AutoCorr( Ser, cols=c, la=lag) to make the gamma matrix
def MakeSumMat(lag):
    rows = (lag)//2   ### Integer division!
    Out = mat(zeros([rows,lag], dtype=int))
    
    for i in range(rows): 
        Out[i,2*i] = 1
        Out[i,2*i+1] = 1
    
    return Out


### Finds the cutting time, when the gammas become negative
def Cutts(Gamma):
    cols = shape(Gamma)[1]
    rows = shape(Gamma)[0]
    Out = mat(zeros([1,cols], dtype=int))
    Stop = mat(zeros([1,cols], dtype=bool))
    
    if (rows == 1):
        return Out
        
    i = 0
    ###while (not(all(Stop)) & (i < (rows-1))):
    for i in range(rows-1):
        for j in range(cols):  ### while Gamma stays positive and decreasing
            if (((Gamma[i+1,j] > 0.0) & (Gamma[i+1,j] < Gamma[i,j])) & (not Stop[0,j])):
                Out[0,j] = i+1 ## the cutting time for colomn j is i+i
            else:
                Stop[0,j] = True
        i += 1
    
    
    return Out


####  Automatically find a maxlag for IAT calculations
def AutoMaxlag( Ser, c, rholimit=0.05, maxmaxlag=20000):
    Co = AutoCov( Ser, c, la=1)
    rho = Co[0,1]/Co[0,0]  ### lag one autocorrelation
    
    ### if autocorrelation is like exp(- lag/lam) then, for lag = 1
    lam = -1.0/log(abs(rho)) 
    
    ### Our initial guess for maxlag is 1.5 times lam (eg. three times the mean life)
    maxlag = int(floor(3.0*lam))+1
    
    ### We take 1% of lam to jump forward and look for the
    ### rholimit threshold
    jmp = int(ceil(0.01*lam)) + 1
    
    T = shape(Ser)[0]  ### Number of rows in the matrix (sample size)

    while ((abs(rho) > rholimit) & (maxlag < min(T//2,maxmaxlag))):
        Co = AutoCov( Ser, c, la=maxlag)
        rho = Co[0,1]/Co[0,0]
        maxlag = maxlag + jmp
        ###print("maxlag=", maxlag, "rho", abs(rho), "\n")
        
    maxlag = int(floor(1.3*maxlag));  #30% more
    
    if (maxlag >= min(T//2,maxmaxlag)): ###not enough data
        fixmaxlag = min(min( T//2, maxlag), maxmaxlag)
        print("AutoMaxlag: Warning: maxlag= %d > min(T//2,maxmaxlag=%d), fixing it to %d" % (maxlag, maxmaxlag, fixmaxlag))
        return fixmaxlag
    
    if (maxlag <= 1):
        fixmaxlag = 10
        print("AutoMaxlag: Warning: maxlag= %d ?!, fixing it to %d" % (maxlag, fixmaxlag))
        return fixmaxlag
        
    print("AutoMaxlag: maxlag= %d." % maxlag)
    return maxlag
    
    
### Find the IAT
def IAT( Ser, cols=-1,  maxlag=0, start=0, end=-1):

    ncols = shape(mat(cols))[1] ## Number of columns to analyse (parameters)
    if ncols == 1:
        if (cols == -1):
            cols = shape(Ser)[1]-1 ### default = last column
        cols = [cols]
    
    if (maxlag == 0):
        for c in cols:
            maxlag = max(maxlag, AutoMaxlag( Ser[start:end,:], c))

    #print("IAT: Maxlag=", maxlag)

    #Ga = MakeSumMat(maxlag) * AutoCorr( Ser[start:end,:], cols=cols, la=maxlag)
    
    Ga = mat(zeros((maxlag//2,ncols)))
    auto = AutoCorr( Ser[start:end,:], cols=cols, la=maxlag)
    
    ### Instead of producing the maxlag/2 X maxlag MakeSumMat matrix, we calculate the gammas like this
    for c in range(ncols):
        for i in range(maxlag//2):
            Ga[i,c] = auto[2*i,c]+auto[2*i+1,c]
    
    cut = Cutts(Ga)
    nrows = shape(Ga)[0]
        
    ncols = shape(cut)[1]
    Out = -1.0*mat(ones( [1,ncols] ))
    
    if any((cut+1) == nrows):
        print("IAT: Warning: Not enough lag to calculate IAT")
    
    for c in range(ncols):
        for i in range(cut[0,c]+1):
            Out[0,c] += 2*Ga[i,c]
    
    return Out


#### Some auxiliar functions and constants for class pytwalk
## square of the norm.
def SqrNorm(x):
    return sum(x*x) 

log2pi = log(2*pi)
log3 = log(3.0)

def Remain( Tr, it, sec1, sec2):
    """ Remaining time Information messages:
        total iterations Tr, current it, start time, current time, as returned by time() (floats)."""

    # how many seconds remaining
    ax = int( (Tr - it) *  ((sec2 - sec1)/it) )


    if (ax < 1):

        return " "

    if (ax < 60):

        return "Finish in approx. %d sec." % (ax,)

    if (ax <= 360):

        return "Finish in approx. %d min and %d sec." % ( ax // 60, ax % 60)

    if (ax > 360):

        ax += sec2  # current time plus seconds remaining=end time
        return "Finish by " + strftime("%a, %d %b %Y, %H:%M.", localtime(ax))



class pytwalk:
    """This is the t-walk class.

    Initiates defining the dimension n, Supp= defines the support
    returns True if x within the support and False otherwise,
    and U= -log of the objective function, eg:

    Mytwalk = pytwalk( n=3, U=MyMinusLogf, Supp=MySupportFunction).

    or
    
    if k positive, u= -log likelihood and w= -log prior 
     penilized likelihood: U(x) = k*u(x) + w(x)

    Mytwalk = pytwalk( n=3, t=0.5, u=MyMinusLoglikelihood, w=MyMinusLogPrior, Supp=MySupportFunction).

    In this case the objective function is U= t*u + w and u, for x (not xp)
    is saved in self.Output_u.  This is a backwards compatible implementation for
    penilized likelihood for thermodynamicl integral to estimate
    normalizing constants.

    Then do: Mytwalk.Run?

    Other parameter are:
    ww= the prob. of choosing each kernel, aw, at, n1phi (see inside twalk.py)
    with default values as in the paper, normally NOT needed to be changed."""

    def __init__( self, n, U=(lambda x: sum(0.5*x**2)), Supp=(lambda x: True),
        k=-1, u=(lambda x: sum(0.5*x**2)), w=(lambda x: 0.0),
        ww=[0.0000, 0.4918, 0.4918, 0.0082, 0.0082], aw=1.5, at=6.0, n1phi=4.0,
        silent=False):
        ### Careful the Hop move does not work!!
        self.n = n
        self.k = k
        if self.k >= 0: ### Penilized likelihood, k*LikelihoodEnergy+PriorEnergy
            self.LikelihoodEnergy = u
            self.PriorEnergy = w
            self.Output_u = array([0.0])
        else:  ### Usual case
            self.PriorEnergy = (lambda x: 0.0) 
            self.LikelihoodEnergy = U
            self.k = 1.0
        self.U = (lambda x: self.Energy(x))
        self.Supp = Supp
        self.Output = zeros((1, n+1)) ### No MCMC output yet
        self.Output_u = array([0.0]) ### To save ll_e, the likelihood energy
        self.T = 1
        self.Acc = zeros(6)  ### To save the acceptance rates of each kernel, and the global acc. rate

        #### Kernel probabilities
        self.Fw = cumsum(ww)
        
        #### Parameters for the propolsals
        self.aw = aw  ### For the walk move
        self.at = at ### For the Traverse move

        #n1phi = 5 ### expected value of parameters to move
        self.pphi = min( n, n1phi)/(1.0*n) ### Prob. of choosing each par.
        
        self.WAIT = 30
        
        self.silent=silent
        
        self.col_names = ""
        self.par_names = []
        for pn in range(self.n):
            self.par_names += ["parameter %d" % (pn)]
            self.col_names += '"par ' + str(pn) + '", '
        self.col_names += '"Energy"' #last colomun, -logpost
        
        self.stop_twalk = False

    def sigint_handler( self, signal, frame):
        print('twalk interrupted, finishing current iteration ... ', end='')
        self.stop_twalk = True

    def Energy( self, x):
        self.ll_e = self.LikelihoodEnergy(x)
        self.prior_e = self.PriorEnergy(x)
        return self.k*self.ll_e + self.prior_e

    def _SetUpInitialValues( self, x0, xp0):
        """Private method."""

        ### Check x0 and xp0 in the support

        if any(abs(x0 -xp0) <= 0):
            print("pytwalk: ERROR, not all entries of initial values different.")
            return [ False, 0.0, 0.0]

        if not(self.Supp(x0)):
            print("pytwalk: ERROR, initial point x0 out of support.")
            return [ False, 0.0, 0.0]
        u = self.U(x0)

        if not(self.Supp(xp0)):
            print("pytwalk: ERROR, initial point xp0 out of support.")
            return [ False, u, 0.0]
        up = self.U(xp0)
        
        return [ True, u, up]



    def Run( self, T, x0, xp0, k=1, thin=1):
        """Run the twalk.
        
           Run( T, x0, xp0, k=1),
           T = Number of iterations.
           x0, xp0, two initial points within the support,
           ***each entry of x0 and xp0 most be different***.
           If either of x0 or xp0 (or both) are None, continue with the
           current values of x and xp.  This is useful to continue a stopped MCMC
           
           If using a penalized likelihhod, change the penalization to k.
           
           To save not all iterations, change thin(=1) to save every thin iterations (thining).
           This is usefule when very big, high dimensional samples, are needed, to save
           memory space.
        """
        
        self.k = k
        
        if not(self.silent):
            sec = time()
            print("pytwalk: Running the twalk with %d iterations"\
                    % (T,), end=' ')
            if self.k == 1:
                print(". ",  strftime("%a, %d %b %Y, %H:%M:%S.", localtime(sec)))
            else:
                print(" (k=%f). " % (self.k,), strftime("%a, %d %b %Y, %H:%M:%S.", localtime(sec)))

        if thin > 1:
            print("pytwalk: Saving every %d iterations only." % (thin))

        if (x0 is None) or (xp0 is None):
            ### Continue with the MCMC, the next line will fail if no previous MCMC was run
            x0 = self.x
            xp0 = self.xp
            ### Add to the array's to place the iterations and the U's ... we donot save up's
            self.Output = append( self.Output, zeros((1 + T//thin, self.n+1)), axis=0)
            self.Output_u = append( self.Output_u, zeros(1 + T//thin))
            T = T + self.T ### Add to the existing number of iterations
            for i in range(6):
                if self.kercall[i] != 0:
                    self.Acc[i] *= self.kercall[i] #Recover back the number of accepted 
            it0 = self.T
        else:
            ### New run, set the array to place the iterations and the U's ... we donot save up's
            self.Output = zeros((1 + T//thin, self.n+1))
            self.Output_u = zeros(1 + T//thin)
            self.T = T+1
            self.Acc = zeros(6)
            self.kercall = zeros(6) ## Times each kernel is called
            it0 = 0
            
        ### Check x0 and xp0 are in the support
        [ rt, u, up] = self._SetUpInitialValues( x0, xp0)

        if (not(rt)):
            return 0
        
        ### send an estimation for the duration of the sampling, since
        ### we have evaluated the ob. func. twice (in self._SetUpInitialValues).
        
        if not(self.silent):
            sec2 = time() # last time we sent a message
            print("       " + Remain( T-it0, 2, sec, sec2))

        x = x0
        xp = xp0
        
        self.Output[ 0, 0:self.n] = x.copy()
        self.Output[ 0, self.n] = u
        self.Output_u[0] = self.ll_e

        j1=1
        j=0

        self.stop_twalk = False # To catch the Keybpoard interrupt
        dfl_handler = signal.signal( signal.SIGINT, self.sigint_handler)
        ### Sampling
        for it in tqdm(range( it0, T - mod((T-it0), thin))):

            y, yp, ke, A, u_prop, up_prop = self.onemove( x, u, xp, up)

            self.kercall[ke] += 1
            self.kercall[5] += 1 
            if (uniform.rvs() < A):  
                x = y.copy()   ### Accept the propolsal y
                u = u_prop
                xp = yp.copy()   ### Accept the propolsal yp
                up = up_prop
                self.Acc[ke] += 1
                self.Acc[5] += 1


            ### To retrive the current values
            self.x = x
            self.xp = xp
            self.u = u
            self.up = up
            if mod((it-it0) , thin) == 0:
                ### Save iteration
                self.Output[1 + (it-it0)//thin,0:self.n] = x.copy()
                self.Output[1 + (it-it0)//thin,self.n] = u
                self.Output_u[1 + (it-it0)//thin] = self.ll_e

            if self.stop_twalk:
                print("done.")
                signal.signal( signal.SIGINT, dfl_handler)
                self.Output = self.Output[:(1 + (it-it0)//thin + 1),:]
                self.Output_u = self.Output_u[:(1 + (it-it0)//thin + 1)]
                self.stop_twalk = False
                break
         
            ### Estimate the remaing time, every 2**j1 iterations
            if not(self.silent):
                if ((it % (1 << j1)) == 0):
                    
                    j1 += 1
                    j1 = min( j1, 10)  # check the time at least every 2^10=1024 iterations
                    ax = time()
                    if ((ax - sec2) > (1 << j)*self.WAIT): # Print an estimation every WAIT*2**j 
                        print("pytwalk: %10d iterations so far. " % (it-it0,) + Remain( T-it0, it-it0, sec, ax))
                        sec2 = ax
                        j += 1
                        j1 -= 1 # check the time as often 

        signal.signal( signal.SIGINT, dfl_handler)
        for i in range(6):
            if self.kercall[i] != 0:
                self.Acc[i] /= self.kercall[i]
        self.T = self.Output.shape[0]
        if not(self.silent):
            if (self.Acc[5] == 0):
                print("pytwalk: WARNING,  all propolsals were rejected!")
                print(strftime("%a, %d %b %Y, %H:%M:%S.", localtime(time())))
                return 0
            else:
                print("pytwalk: finished, %d iterations." % (self.T*thin,)) 
                print(strftime("%a, %d %b %Y, %H:%M:%S.", localtime(time())))

        return 1


    def  onemove( self, x, u, xp, up):
        """One move of the twalk.  This is basically the raw twalk kernel.
           It is usefull if the twalk is needed inside a more complex MCMC.
        
           onemove(x, u, xp, up),
           x, xp, two points WITHIN the support ***each entry of x0 and xp0 must be different***.
           and the value of the objective at x, and xp
           u=U(x), up=U(xp).
           
           It returns: [y, yp, ke, A, u_prop, up_prop]
           y, yp: the proposed jump
           ke: The kernel used, 0=nothing, 1=Walk, 2=Traverse, 3=Blow, 4=Hop
           A: the M-H ratio
           u_prop, up_prop: The values for the objective func. at the proposed jumps 
        """

        #### Make local references for less writing
        n = self.n
        U = self.U
        Supp = self.Supp
        Fw = self.Fw
        
        ker = uniform.rvs() ### To choose the kernel to be used
        ke = 1
        A = 0
        
        ## Kernel nothing exchange x with xp, not used
        if ((0.0 <= ker) & (ker < Fw[0])): 
            ke = 0
            y = xp.copy()
            up_prop = u
            yp = x.copy()
            u_prop = up
            ### A is the MH acceptance ratio
            A = 1.0;  #always accepted


        ## The Walk move
        if ((Fw[0] <= ker) & (ker < Fw[1])):
            
            ke = 1

            dir = uniform.rvs()

            if ((0 <= dir) & (dir < 0.5)):  ## x as pivot
        
                yp = self.SimWalk( xp, x)

                y = x.copy()
                u_prop = u

                if ((Supp(yp)) & (all(abs(yp - y) > 0))):
                    up_prop = U(yp)
                    A = exp(up - up_prop)
                else:
                    up_prop = None
                    A = 0; ##out of support, not accepted
                        
            else:  ## xp as pivot

                y = self.SimWalk( x, xp)

                yp = xp.copy()
                up_prop = up

                if ((Supp(y)) & (all(abs(yp - y) > 0))):
                    u_prop = U(y)
                    A = exp(u - u_prop)
                else:
                    u_prop = None
                    A = 0; ##out of support, not accepted


        #### The Traverse move
        if ((Fw[1] <= ker) & (ker < Fw[2])):

            ke = 2
            dir = uniform.rvs()

            if ((0 <= dir) & (dir < 0.5)):  ## x as pivot

                beta = self.Simbeta()
                yp = self.SimTraverse( xp, x, beta)

                y = x.copy()
                u_prop = u
                
                if Supp(yp):                
                    up_prop = U(yp)
                    if (self.nphi == 0):
                        A = 1 ###Nothing moved
                    else:
                        A = exp((up - up_prop) +  (self.nphi-2)*log(beta))
                else:
                    up_prop = None
                    A = 0 ##out of support, not accepted
            else:            ## xp as pivot

                beta = self.Simbeta()
                y = self.SimTraverse( x, xp, beta)

                yp = xp.copy()
                up_prop = up

                if Supp(y):
                    u_prop = U(y)
                    if (self.nphi == 0):
                        A = 1 ###Nothing moved
                    else:
                        A = exp((u - u_prop) +  (self.nphi-2)*log(beta))
                else:
                    u_prop = None
                    A = 0 ##out of support, not accepted

        ### The Blow move
        if ((Fw[2] <= ker) & (ker < Fw[3])): 

            ke = 3
            dir = uniform.rvs()

            if ((0 <= dir) & (dir < 0.5)):  ## x as pivot
                yp = self.SimBlow( xp, x)
                
                y = x.copy()
                u_prop = u
                if ((Supp(yp)) & all(yp != x)):
                    up_prop = U(yp)
                    W1 = self.GBlowU( yp, xp,  x)
                    W2 = self.GBlowU( xp, yp,  x) 
                    A = exp((up - up_prop) + (W1 - W2))
                else:
                    up_prop = None
                    A = 0 ##out of support, not accepted
            else:  ## xp as pivot
                y = self.SimBlow( x, xp)

                yp = xp.copy()
                up_prop = up
                if ((Supp(y)) & all(y != xp)):
                    u_prop = U(y)
                    W1 = self.GBlowU(  y,  x, xp)
                    W2 = self.GBlowU(  x,  y, xp)
                    A = exp((u - u_prop) + (W1 - W2))
                else:
                    u_prop = None
                    A = 0 ##out of support, not accepted
        

        ### The Hop move
        if ((Fw[3] <= ker) & (ker < Fw[4])): 

            ke = 4
            dir = uniform.rvs()

            if ((0 <= dir) & (dir < 0.5)):  ## x as pivot
                yp = self.SimHop( xp, x)
                
                y = x.copy()
                u_prop = u
                if ((Supp(yp)) & all(yp != x)):
                    up_prop = U(yp)
                    W1 = self.GHopU( yp, xp,  x)
                    W2 = self.GHopU( xp, yp,  x) 
                    A = exp((up - up_prop) + (W1 - W2))
                else:
                    up_prop = None
                    A = 0 ##out of support, not accepted
            else:  ## xp as pivot
                y = self.SimHop( x, xp)

                yp = xp.copy()
                up_prop = up
                if ((Supp(y)) & all(y != xp)):
                    u_prop = U(y)
                    W1 = self.GHopU(  y,  x, xp)
                    W2 = self.GHopU(  x,  y, xp)
                    A = exp((u - u_prop) + (W1 - W2))
                else:
                    u_prop = None
                    A = 0 ##out of support, not accepted
        
        return [y, yp, ke, A, u_prop, up_prop]



    #################################################################################
    ##### Auxiliar methods for the 4 kernels

    ### Used by the Walk kernel
    def SimWalk( self, x, xp):
        aw = self.aw
        n = self.n
        
        phi = (uniform.rvs(size=n) < self.pphi) ### parametrs to move
        self.nphi = sum(phi)
        z = zeros(n)

        for i in range(n):
            if phi[i]:
                u = uniform.rvs()
                z[i] = (aw/(1+aw))*(aw*u**2.0 + 2.0*u - 1.0)

        return x + (x - xp)*z

    #### Used by the Traverse kernel
    def Simbeta(self):
        at = self.at
        if (uniform.rvs() < (at-1.0)/(2.0*at)):
            return exp(1.0/(at+1.0)*log(uniform.rvs()))
        else:
            return exp(1.0/(1.0-at)*log(uniform.rvs()))

    def SimTraverse( self,  x, xp, beta):
        n = self.n
    
        phi = (uniform.rvs(size=n) < self.pphi)
        self.nphi = sum(phi)

        rt = x.copy()
        for i in range(n):
            if (phi[i]):
                rt[i] = xp[i] + beta*(xp[i] - x[i])
            
        return rt


    ### Used by the Blow kernel
    def SimBlow( self, x, xp):
        n = self.n
    
        self.phi = (uniform.rvs(size=n) < self.pphi)
        self.nphi = sum(self.phi)
    
        self.sigma = max(self.phi*abs(xp - x))

        rt = x.copy()
        for i in range(n):
            if (self.phi[i]):
                rt[i] = xp[i] + self.sigma * norm.rvs()
            
        return rt


    def GBlowU( self, h, x, xp):
        nphi = self.nphi
        self.sigma = max(self.phi*abs(xp - x)) #recalculate sigma, but same phi    
        if (nphi > 0):
            return (nphi/2.0)*log2pi + nphi*log(self.sigma) + 0.5*SqrNorm(h - xp)/(self.sigma**2)
        else: 
            return 0


    ### Used by the Hop kernel
    def SimHop( self, x, xp):
        n = self.n
    
        self.phi = (uniform.rvs(size=n) < self.pphi)
        self.nphi = sum(self.phi)
    
        self.sigma = max(self.phi*abs(xp - x))/3.0

        rt = x.copy()
        for i in range(n):
            if (self.phi[i]): 
                rt[i] = x[i] + self.sigma * norm.rvs()

        return rt


    def GHopU( self, h, x, xp): ## It is actually equal to GBlowU!
        nphi = self.nphi
        self.sigma = max(self.phi*abs(xp - x))/3.0 ##Recalculate sigma, but same phi

        if (nphi > 0): #Mistake until 20AUG2020, formely pivot was left to xp: 
            return (nphi/2.0)*log2pi + nphi*log(self.sigma) + 0.5*SqrNorm(h - x)/(self.sigma**2) #
        else: 
            return 0



#################################################################################
#####  Output analysis auxiliar methods

    def IAT( self, par=-1, burn_in=0, start=0, end=-1, maxlag=0):
        """Calculate the Integrated Autocorrelation Times of parameters par
           the default value par=-1 is for the IAT of the U's"""
        
        iat = IAT( self.Output, cols=par, maxlag=maxlag, start=start, end=end)
        
        return iat
    

    def PlotTs( self, par=-1, burn_in=0, iat=1, end=-1, ax=None, ylabel=None, **kwargs):
        """Plot a time series of parameter (number) par. par may also be the (partial) name
           of the parameter (string).
           par=-1 plots the logpost (not the Energy). 
           The plot is done in the axis ax, if None, one is created and returned.
           All other kargs are passed to ax.plot .
        """
        if isinstance(par,str):
            tmp = self.fpi(par)
            if tmp == -1:
                print("PyPstwalk:PlotPost: par. name %s not found." % (par))
                return None
            elif tmp < -1:
                print("PyPstwalk:PlotPost: %d matches found for par. name %s." % (-tmp-1,par))
                return None
            else:
                par = tmp # par index
        if ax is None:
            fig, ax = subplots()
        if end == -1:
            end = self.T
        t = arange( burn_in, end, iat)
        mult = 1
        if par == -1:
            ylabel = "LogPost"
            mult = -1
        elif ylabel is None:
            ylabel = self.par_names[par]
        ax.plot( t, mult*self.Output[burn_in:end:iat,par], '-', **kwargs)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        return ax

    def TS( self, par=-1, start=0, end=0):
        """This is a legacy method, now use PlotTs, use burn_in instead of start. 
           Plot a time series of parameter (number) par. par may also be the (partial) name
           of the parameter (string).
           par=-1 plots the logpost (not the Energy). 
           The plot is done in the axis ax, if None, one is created and returned.
           All other kargs are passed to ax.plot .
           If burn_in = -1, self.default_burn_in is used.
           
           returns the axes used.
        """
        ax = self.PlotTs( par=par, burn_in=start, end=end)
        return ax

    def Ana( self, par=-1, burn_in=0, start=-1, end=-1):
        """Output Analysis, TS plots, accepatnce rates, IAT etc.
            start left as legacy (if start > 0, burn_in=start),
            use burn_in instead.
        """
        
        if start > 0:
            burn_in = start
        
        print("Acceptance rates for the Walk, Traverse, Blow and Hop kernels:" + str(self.Acc[1:5]))
        print("Global acceptance rate: %7.5f" % self.Acc[5])
        
        self.iat = int(ceil(self.IAT( par=par, burn_in=burn_in, end=end)))
        print("Integrated Autocorrelation Time: %d, IAT/n: %7.1f" % (self.iat, self.iat/self.n))
        
        self.PlotTs( par=par, burn_in=burn_in, end=end)
        return self.iat


    def PlotMarg( self, par=-1, burn_in=0, start=-1, end=-1, iat=1, g=(lambda x: x[0]),\
                 ax=None, xlabel=None, density=True, **kwargs):
        """
           Basic histograms of 1D marginal distributions.
           If par=-1, use g.
           The function g provides a transformation to be applied to the data,
           eg g=(lambda x: abs(x[0]-x[1]) would plot a histogram of the distance
           between parameters 0 and 1, etc.
           
           ax: axes to use, if None, creates one.
           xlabel: xlabel to use, if None, use par number
           
           returns the axes used.
         """
        if start > 0:
             burn_in = start
        if end == -1:
            end = self.T

        if (par == -1):
            indx = arange( burn_in, end, step=iat)
            ser = zeros(indx.size)
            for i,it in enumerate(indx):
                x = self.Output[it, :-1]
                ser[i] = g(x)
            if (xlabel == None):
                xlabel = "g"
        else:
            ser = self.Output[burn_in:end:iat, par]
            if (xlabel == None):
                xlabel = self.par_names[par]
            
        if ax is None:
            fig, ax = subplots()
        ax.hist( ser, density=density, **kwargs)
        ax.set_xlabel(xlabel)
        if density:
            ax.set_ylabel("density")

        return ax

    def Hist( self, par=-1, burn_in=0, start=-1, end=-1, iat=1, g=(lambda x: x[0]),\
                 ax=None, xlabel=None, density=True, **kwargs):
        """
           This is a legacy method, use PlotMarg instead.
           
           Basic histograms of 1D marginal distributions.
           If par=-1, use g.
           The function g provides a transformation to be applied to the data,
           eg g=(lambda x: abs(x[0]-x[1]) would plot a histogram of the distance
           between parameters 0 and 1, etc.
           
           ax: axes to use, if None, creates one.
           xlabel: xlabel to use, if None, use par number
           
           returns the axes used.
         """
        return self.PlotMarg( par=par, burn_in=burn_in, start=start, end=end, iat=iat, g=g,\
                     ax=ax, xlabel=xlabel, density=density, **kwargs)

    def SavetwalkOutput(self, fnam, burn_in=0, end=-1, iat=1, colnames=None):
        """Save the twalk Output as a csv file."""
        savetxt( fnam, self.Output[burn_in:end:iat,:], fmt='%.15g', comments='#',\
            header="### pytwalk: twalk Output, finished: %s\n" % (strftime( "%Y.%m.%d:%H:%M:%S", localtime()))\
                     + self.col_names)
        print("pytwalk: twalk output saved in %s." % (fnam))
    
    def LoadtwalkOutput(self, fnam, skiprows=1, **kwargs):
        """Load the twalk Output."""
        self.Output = loadtxt(fnam, skiprows=1, **kwargs)
        self.T = self.Output.shape[0]

     
    def PlotCorner( self, pars=None, burn_in=0, density=True, **kwargs):
        """Make a corner plot of the list of parameters pars.  If pars=None the plot
           all parameters returns the figure instance. 
        """
        if pars is None:
            pars = arange(self.n)
        pars = array(pars)
        fig = corner(self.Output[burn_in:,pars], hist_kwargs={'density':density}, **kwargs)
        fig.tight_layout()
        return fig

        
    ##### A simple Random Walk M-H
    def RunRWMH( self, T, x0, sigma):
        """Run a simple Random Walk M-H"""

        sec = time() # last time we sent a message
        print("pytwalk: This is the Random Walk M-H running with %d iterations." % T)
        ### Local variables
        x = x0.copy()
        if not(self.Supp(x)):
            print("pytwalk: ERROR, initial point x0 out of support.")
            return 0
        self.T = T

        u = self.U(x)
        n = self.n

        sec2 = time() # last time we sent a message
        print("       " + Remain( T, 2, sec, sec2))

        ### Set the array to place the iterations and the U's
        self.Output = zeros((T+1, n+1))
        self.Acc = zeros(6)
                
        #### Make local references for less writing
        Output = self.Output
        U = self.U
        Supp = self.Supp
        Acc = self.Acc
        
        Output[ 0, 0:n] = x.copy()
        Output[ 0, n] = u

        j1=1
        j=0

        y = x.copy()
        for it in range(T):
            y = x + norm.rvs(size=n)*sigma ### each entry with sigma[i] variance 
            if Supp(y):        ### If it is within the support of the objective
                uprop = U(y)   ### Evaluate the objective
                if (uniform.rvs() < exp(u-uprop)):  
                    x = y.copy()   ### Accept the propolsal y
                    u = uprop
                    Acc[5] += 1

            ### Estimate the remaing time, every 2**j1 iterations
            if ((it % (1 << j1)) == 0):

                j1 += 1
                j1 = min( j1, 10)  # check the time at least every 2^10=1024 iterations
                ax = time()
                if ((ax - sec2) > (1 << j)*self.WAIT): # Print an estimation every WAIT*2**j 
                    print("pytwalk: %10d iterations so far. " % (it,) + Remain( T, it, sec, ax))
                    sec2 = ax
                    j += 1
                    j1 -= 1 # check the time as often 

            Output[it+1,0:n] = x
            Output[it+1,n] = u
        
        if (Acc[5] == 0):
            print("pytwalk: WARNING,  all propolsals were rejected!")
            return 0

        Acc[5] /= T;
        return 1





"""
##############################################################3
Created on Tue Oct 11 13:15:17 2022

@author: J Andrés Christen, jac at cimat.mx

Posterior sampling using the twalk:
Templete class, pyPstwalk, to perform Bayesian inference using the t-walk.

Use derived classes to perform specific inferences.See pytwalktutorial.py
"""

class pyPstwalk(pytwalk):
    """
        pyPstwalk: pyThon class to perform Psterior sampling (for Bayesian
        inference) using the t-walk:
        
        This is a template class, a derived class needs to be coded to define
        the loglikehood.  See below the examples for independent 1D sampling,
        ind1Dsampl, and for Bernoulli-Weibull regresion .
        
        par_names: a list with the parameter names, in the correct order.
           Internally, an array will hold the values for this parameters and
           this defines in which order; the same for par_prior and par_supp.
        par_prior: a list with objects definig the marginal prior of each
           parameter.  logprior assumes independence, and this may be changed
           by overloading this method.  The objects defining the priors most
           have the method logpdf. Typically a 'frozen' distribution from
           scipy.stats (see examples below).  Also, it should have the method
           rvs to simulate initial values, from the prior, for the t-walk.
        par_supp: a list of callables, taking a float and returning True or False
           to define if the corresponding parameters is its support.
        default_burn_in: a default vlaue for the burn_in.
        default_op_fnam: a default name for a csv file to save the twalk.Output
           array to.  This is automatically done after the t-walk finishes and
           may be retrived with LoadtwalkOuput.
        
        The likelihood may be penalized with k (=1):
        u= -log likelihood and w= -log prior 
         penilized likelihood: U(x) = k*u(x) + w(x)
        The -log likelihood value of each iteration is saved in Output_u 
    """
    
    def __init__(self, par_names, par_prior, par_supp, default_burn_in=0, k=1):
        
        self.q = len(par_names) #Number of parameters
        super().__init__(  n=self.q, U=self.Energy, Supp=self.Supp, k=k)
        self.k = k #No penalization
        self.par_names = par_names # Overwrite the default par names
        
        ### Create a string with the par names separated by commas,
        ### for the csv file to save twalk Output
        self.col_names = ""
        for pn in self.par_names:
            self.col_names += '"' + pn + '", '
        self.col_names += '"Energy"' #last colomun, -logpost
        ###                 a                      b                 al              beta
        self.par_prior = par_prior        
        self.par_supp  = par_supp
        self.default_burn_in = default_burn_in
        self.par_true = None
        
    def fpi( self, pname):
        """Find the index of parameter with (possible partial) name pname.
           returns -1 if no matches, < 1 if several matches or
             the index if only one match.
        """
        tmp = where([pn.count(pname)>0 for pn in self.par_names])[0]
        if tmp.size != 1:
            ### pnam not found or several matches found
            return -tmp.size -1 #-1 no matches, < .1 several matches
        else:
            return tmp[0]
    
    def loglikelihood(self, x):
        """Overload this method to define the likelihood.
             x is an array containing the parameters in the order spacified
             by self.par_names.  One may translate this first to define the
             loglikelihood, e.g:
            a, b, c = x
        """
        pass
    
    def logprior(self, x):
        """The log prior assumes independece and multiplies the marginal prior
            for each parameter.
            Overload this method if a more complex, non-independant, prior is needed.
        """
        return sum([prior.logpdf(x[i]) for i,prior in enumerate(self.par_prior)])
    
    def Supp(self, x):
        """Checks if all parameters are in the support, evaluating par_supp.
           Overload this method if a more complex, non-independant, support is needed.
        """
        return all([supp(x[i]) for i,supp in enumerate(self.par_supp)])
    
    def SimPrior(self):
        """Simulated from the prior.
           Overload this method if a more complex, non-independant, prior is needed.
        """
        return array([prior.rvs() for prior in self.par_prior])
    
    def SimInit(self):
        """Simulates an initial value for the t-walk from the prior."""
        return self.SimPrior()
    
    def Energy( self, x):
        """-log of the posterior.  The likelihood may be penalized with self.k."""
        self.ll_e = -1*self.loglikelihood(x)
        self.prior_e = -1*self.logprior(x)
        return self.k*self.ll_e + self.prior_e
        
    def RunMCMC( self, T, burn_in=0, fnam=None, **kwargs):
        """Runs the t-walk for T iterations.  Ana is called to calculate
             the IAT with a burn in of burn_in.
             fnam is an optional file name to save twalk Output to.
        """
        self.Run( T=T, x0=self.SimInit(), xp0=self.SimInit(), **kwargs)
        if burn_in == 0:
            print("Use burn_in > 0 to run the sample analysis Ana automatically.")
            self.PlotTs()
            print("Now run: self.Ana(burn_in=??).")
            iat = 1
        else:
            iat = self.Ana(burn_in=burn_in)
        if fnam is not None:
            self.SavetwalkOutput(fnam)
        return iat
        
    def PlotPrior( self, par, qm=1e-6, qM=1- 1e-6, ax=None, color='green', **kwargs):
        """Plot prior. Add to existing plot within its x limits if ax is not None.
           If ax=None, par_prior requires the quantile mathod ppf,
           ploting from the qm=1e-6 to qM=1-1e-6 quantiles.
           returns the axes.           
        """
        if isinstance(par,str):
            tmp = self.fpi(par)
            if tmp == -1:
                print("PyPstwalk:PlotPost: par. name %s not found." % (par))
                return None
            elif tmp < -1:
                print("PyPstwalk:PlotPost: %d matches found for par. name %s." % (-tmp-1,par))
                return None
            else:
                par = tmp # par index
        if ax is None:
            fig, ax = subplots()
            xl = [self.par_prior[par].ppf(qm), self.par_prior[par].ppf(qM)]
        else:
            xl = ax.get_xlim()
        x = linspace( xl[0], xl[1], num=200)
        ax.plot( x, exp(self.par_prior[par].logpdf(x)), '-', color=color, **kwargs)
        ax.set_xlim(xl)
        return ax            

    def PlotPost( self, par, burn_in=0, ax=None,\
                 prior_color="green", density=True, **kwargs):
        """Plot a histogram of parameter (number) par. par may also be the (partial) name
           of the parameter (string).
           The plot is done in the axis ax, if ax=None, one is created and returned.
           All other kargs are passed to ax.hist .
           If burn_in = -1, self.default_burn_in is used.
           
           The prior is also plotted within the bounds of the posterior (it may
            appear as a line only etc.), with prior_color.  If prior_color=None
           then the prior is not plotted.
        """
        if isinstance(par,str):
            tmp = self.fpi(par)
            if tmp == -1:
                print("PyPstwalk:PlotPost: par. name %s not found." % (par))
                return None
            elif tmp < -1:
                print("PyPstwalk:PlotPost: %d matches found for par. name %s." % (-tmp-1,par))
                return None
            else:
                par = tmp # par index
        if ax is None:
            fig, ax = subplots()
        ax.hist( self.Output[burn_in:,par], density=density, **kwargs)
        if self.par_true is not None:
            ax.axvline( self.par_true[par], ymax=0.1, color="black") #alpha true value
        ax.set_xlabel(self.par_names[par])
        if prior_color is not None:
            self.PlotPrior( par=par, color=prior_color, ax=ax)
        return ax

     
    def PlotCorner( self, pars=None, burn_in=0, density=True, prior_color='green', **kwargs):
        """Make a corner plot of the list of parameters pars.  The list may be
           by (partial) parameter name or parameter number. If pars=None the plot
           all parameters
           returns the figure instance. 
        """
        if pars is None:
            pars = arange(self.q)
        for i,par in enumerate(pars):
            if isinstance(par,str):
                tmp = self.fpi(par)
                if tmp == -1:
                    print("PyPstwalk:PlotCorner: par. name %s not found." % (par))
                    return None
                elif tmp < -1:
                    print("PyPstwalk:PlotCorner: %d matches found for par. name %s." % (-tmp-1,par))
                    return None
                else:
                    pars[i] = tmp # par index
        pars = array(pars)
        fig = corner(self.Output[burn_in:,pars], hist_kwargs={'density':density}, **kwargs)
        axes = array(fig.axes).reshape((pars.size, pars.size))
        if prior_color is not None:
            for i in range(pars.size):
                self.PlotPrior( par=pars[i], color=prior_color, ax=axes[i,i])
                axes[-1,i].set_xlabel(self.par_names[pars[i]])
        if self.par_true is not None:
            for i in range(pars.size):
                axes[i,i].axvline( self.par_true[pars[i]], ymax=0.1, color="black") #alpha true value
        fig.tight_layout()
        return fig

class Ind1Dsampl(pyPstwalk):
    """Dereived class from pyPstwalk for 1D independent sampling.
         q: numper of unknown parameters.
         data: array with data (=None run with no data).
         logdensity: log of the density, with signature logdensity( data, x),
          evaluates the log of the density at data (needs to be vectorized) for
          parameter x.  See Gamma Sampling example below.
         par_names, par_prior, par_supp, default_op_fnam: as in pyPstwalk
         simdata: optional, to simulate data with signature simdata( n, x), to
          simulate a sample of size n with 'true' parameters x.
    
    """
    
    def __init__( self, q, data, logdensity, par_names, par_prior, par_supp, simdata=None):
        
        self.logdensity = logdensity
        self.data = data
        if data is None:
            self.smpl_size = 0
        else:
            self.smpl_size = self.data.size
        self.simdata = simdata
        
        super().__init__(par_names=par_names,\
                         par_prior=par_prior,\
                         par_supp =par_supp)
            
    def loglikelihood(self, x):
        if self.smpl_size == 0:
            return 0.0 ## Run with no data, ie simulate from the prior
        else:
            return np_sum( self.logdensity( self.data, x))
    
    def SimData(self, n, x):
        self.par_true = x
        self.data = self.simdata( n, x)
        self.smpl_size = self.data.size



class BUQ(pyPstwalk):
    """Dereived class from pyPstwalk for simple Bayesian Inverse problems (Bayesian UQ).
       Or in fact, non-linear regression.
       F_\theta is the Forward Map, or the regresso function.
       $$
        E(y_i | \theta) = F_\theta(t_j)
        y_i | \theta, \simga  \sim G(\sigma)
       $$
       independent.
         q: number of unknown parameters, card of theta + sigma,
            if sigma (dispersion parameters) also unknown.
         sigma: values for the disperssion parameter if known, if sigma=None,
          then it is assumed not known and is the last parameter in the parameter
          list, ie: "x = theta + sigma"
         data: array with data (=None run with no data).
         logdensity: log of the density of G, with signature
            logdensity( data, loc, scale),
          evaluates the log of the density at data (needs to be vectorized) for
          parameter x.
            simdata( n, loc, scale), to
          simulate a sample of size n with 'true' parameters loc=theta and scale=sigma.
          
        F: forward map, with signature F( theta, t), vectorized in t.
        t: localtions (eg. time) to evaluate the Forwad map.

        par_names, par_prior, par_supp, default_op_fnam: as in pyPstwalk
         simdata: optional, to simulate data from G with signature
    
    """
    
    def __init__( self, q, data, logdensity, sigma, F, t,\
                    par_names, par_prior, par_supp, simdata=None):
        
        self.logdensity = logdensity
        self.t = t
        self.data = data
        if data is None:
            self.smpl_size = 0
        else:
            self.smpl_size = self.data.shape
        self.simdata = simdata
        if sigma is None:
            self.sigma_known = False
        else:
            self.sigma_known = True
        self.sig = sigma #internally we use sig, to avoid a clash with pytwalk
        self.F = F
        
        super().__init__(par_names=par_names,\
                         par_prior=par_prior,\
                         par_supp =par_supp)
            
    def loglikelihood(self, x):
        if self.smpl_size == 0:
            return 0.0 ## Run with no data, ie simulate from the prior
        else:
            if self.sigma_known:
                return np_sum( self.logdensity( self.data, loc=self.F( x, self.t), scale=self.sig))
            else:
                return np_sum( self.logdensity( self.data, loc=self.F( x[:-1], self.t), scale=x[-1]))
    
    def SimData(self, x):
        """Simulate data at points self.t"""
        self.par_true = x
        if self.sigma_known:
            #theta = x
            self.data = self.simdata( self.t.shape, loc=self.F( x, self.t), scale=self.sig)
        else:
            #theta = x[:-1], sigma = x[-1]
            self.data = self.simdata( self.t.shape, loc=self.F( x[:-1], self.t), scale=x[-1])
        self.smpl_size = self.data.shape

