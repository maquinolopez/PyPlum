###################################################
###  This is a python implementation of Plum    ###
###  By Marco A. Aquino-Lopez                   ###
###  cite:  Aquino-Lopez, et al. (2018)         ###
###  DOI: 10.1007/s13253-018-0328-7             ###
###################################################

import cProfile
import sys
try:
    from numpy import seterr,ogrid, newaxis, arange, triu, ones, tril, identity,median, delete, logical_and,nditer,r_, sort, append, concatenate, repeat, linspace, interp, genfromtxt, array, exp, log, sum,  savetxt, mean, matrix, sqrt, zeros, cumsum, row_stack,hstack

    from numpy.random import seed, randint
    seterr(all='ignore')
except ImportError:
    print("you need to install Numpy")
    sys.exit(1)
try:
    from scipy.stats import uniform, gaussian_kde, gamma, beta
    from scipy.interpolate import interp1d
except ImportError:
    print ("you need to install SciPy")
    sys.exit(1)
try:
    from matplotlib.pyplot import rc, Line2D, GridSpec, plot, close, show, savefig, hist, xlabel, ylabel, title, axis, subplot, figure, setp, fill_betweenx
except ImportError:
    print ("you need to install Matplotlib")
    sys.exit(1)
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    print("you are missing sklearn module\nPlum won't be able to infer the how many samples\nwill be used for estimating supported 210Pb" )
try:
    import pytwalk
except ImportError:
    print ("you need to install pytwalk\nPlease visit: https://www.cimat.mx/~jac/twalk/")
    sys.exit(1)
import os.path
from time import strftime

class Plum:
    def __init__(self,Core='HP1C',dirt="/Documents/PyPlum/",Dircc="/Documents/PyPlum/Calibration Curves/",
                thick=1.,n_supp=True,mean_m=.4,shape_m=10.,mean_acc=10,shape_acc=1.5,fi_mean=100., fi_shape=1.5,
                s_mean=10,s_shape=1.5,intv=.95,Ts_mod=True,iterations=1500,burnin=4000,thi=25,cc=True,
                ccpb="NONE",tparam=False,showchrono=False,reservoir_eff=False,r_effect_prior=0.,r_effect_psd=500.,
                g_thi=2,Sdate=True,Al=.1,seed=True,d_by=1.):
        self.hfol	        =   os.path.expanduser("~")
        # Define seeds
        if seed:
            self.seeds      =   randint(1000,9000)
        else:
            self.seeds      =   int(seed)
        # Directories and files
        self.Core           =   Core
        self.dirt           =   dirt
        #Model parameters
        self.d_by           =   d_by
        self.shape1_m       =   shape_m   # Shape of memory parameter
        self.mean_m         =   mean_m     # Mean of memory parameter
        self.shape_acc      =   shape_acc  # Shape of accumulation rate
        self.mean_acc       =   mean_acc   # Mean of accumulation rate
        self.fi_shape       =   fi_shape
        self.s_shape        =   s_shape
        self.fi_scale       =   fi_mean/fi_shape
        self.fi_mean        =   fi_mean
        self.s_scale        =   s_mean/s_shape
        self.s_mean         =   s_mean
        self.shape2_m       =   (self.shape1_m*(1-mean_m))/mean_m
        self.scale_acc      =   mean_acc/shape_acc
        # MCMC parameters
        self.iterations     =   iterations
        # Extra parameters
        self.tparam         =   tparam     # True: simulate alphas, False: simulates ms
        self.intv           =   intv       # Intervals
        # Load data
        self.dirt           =   dirt
        self.Sdate          =   Sdate
        self.load_data()
        # Load Calibration Curve
        if self.data_data:
            self.cc             =   cc
            self.ccpb           =   ccpb
            self.Dircc          =   Dircc
            self.load_calcurve()
        # setting bacon sections
        self.by             =   thick         # Thickness of bacon seccion
        self.def_breaks()
        self.m              =   len(self.breaks ) - 1 # Number of sections
        # setting other variables
        self.intv           =   (1-intv)/2.           #
        self.thi            =   int((self.m+2))*thi   # 100 #Testing value
        self.burnin         =   burnin*(self.m+5)     # 20000 #Testing value
        #filename and constants
        self.lam            =   0.03114
        self.pdfname        =   "Chronology_{}_{}_obj.pdf".format(self.Core, self.m)
        # defines which model to used
        self.Ts_mod         =   Ts_mod
        self.reservoir_eff  =   reservoir_eff
        self.tparam         =   tparam
        self.define_model()
        #define other variables
        self.showchrono     =   showchrono
        self.g_thi          =   g_thi
        self.logby          =   log(1./self.by)
        self.iby            =   1/self.by
        self.r_effect_sd    =   1/(2. * (r_effect_psd**2) )
        self.r_effect_prior =   r_effect_prior
        self.matrixone      =   ones([self.m-1,self.m-1])
        tmp_matrix          =   ones([self.m,self.m])
        self.rows, column_indices = ogrid[:self.matrixone.shape[0], :self.matrixone.shape[1]]
        r                   =  -array(range(self.m-1))[::-1] - 1
        r[r < 0]            += self.matrixone.shape[1]
        self.column_indices = column_indices - r[:, newaxis]
        self.Al             = 1/(self.lam*Al)

    def def_breaks(self):
        if self.lead_data:
            self.breaks         =   array(arange(0,max(self.max_pd,self.max_data,self.max_date)+2*self.by,self.by))
        else:
            self.breaks         =   array(arange(min(self.min_data,self.min_date),max(self.max_data,self.max_date)+2*self.by,self.by))

    def load_calcurve(self):
        if self.cc == True:
            if self.data[0,-1] == 1:
                self.cc = "IntCal13.14C"
            elif self.data[0,-1] == 2:
                self.cc = 'Marine13.14C'
            elif self.data[0,-1] == 3:
                self.cc = 'SHCal13.14C'
        if os.path.isfile(self.hfol + self.Dircc + self.cc) :
            intcal              =   genfromtxt(self.hfol + self.Dircc + self.cc, delimiter=',')
            self.ic             =   intcal
            self.cc_mean        =   interp1d(self.ic[:,0],self.ic[:,1], fill_value="extrapolate")
            self.cc_var         =   interp1d(self.ic[:,0],self.ic[:,2], fill_value="extrapolate")
        else:
            print('Please add calibration curves in folder {}'.format(self.Dircc))
            sys.exit(1)
        if array(self.data[:,0]<0).sum() != 0:
            if self.ccpb == 1:
                self.ccpb ='postbomb_NH1.14C'
            elif self.ccpb == 2:
                self.ccpb = 'postbomb_NH2.14C'
            elif self.ccpb == 3:
                self.ccpb = 'postbomb_NH3.14C'
            elif self.ccpb == 4:
                self.ccpb = 'postbomb_SH3.14C'
            elif self.ccpb == 5:
                self.ccpb = 'postbomb_SH1-2.14C'
            if os.path.isfile(self.hfol + self.Dircc + self.ccpb) :
                intcalpost          =   genfromtxt(self.hfol + self.Dircc + self.ccpb, delimiter=",")
                self.ic             =   concatenate((intcalpost, self.ic), axis=0)
                self.cc_mean        =   interp1d(self.ic[:,0],self.ic[:,1],fill_value="extrapolate")
                self.cc_var         =   interp1d(self.ic[:,0],self.ic[:,2], fill_value="extrapolate")
            else:
                print('Please add postbomb calibration curves in folder {}'.format(self.Dircc))
                sys.exit(1)

    def load_data(self):
        #load 210Pb
        if os.path.isfile(self.hfol + self.dirt + '/' + self.Core + '/' + self.Core + '.csv') :
            Data              =   genfromtxt(self.hfol + self.dirt + '/' + self.Core + '/' + self.Core + '.csv', delimiter=',')
            Data              =   Data[1:,1:] ;self.Data=Data               # 210Pb data
            self.depths       =   array(Data[ :,[4,0]])
            self.depths[:,0]  =   self.depths[:,1] - self.depths[:,0]
            self.act          =   array(Data[ :,[2,3]])
            # Preanalysis
            if array(Data).shape[1]     == 6 :        # Agregar la linear regression
                if Data[1,5] > 0:
                    nsupp   = len(Data[:,5])-int(Data[1,5])
                else:
                    nsupp   = self.linreg_supp()
                self.supp         = array(Data[nsupp:,[2,3]])
                ## Poner codigo para cuando esta variable no esta
                self.act          = self.act[0:nsupp,:]
                self.depths       = self.depths[0:nsupp,:]
                self.density      = Data[0:nsupp,1] * 10.
                self.Sdate        = float(Data[0,5])
                self.s_len        = 1
                self.lead_data    = True
                self.max_pd       = max(self.depths[-1,:])
            elif array(Data).shape[1]   == 8 :
                self.supp         = array(Data[:,[5,6]])
                self.density      = Data[:,1] * 10.
                self.Sdate        = float(Data[0,7])
                self.s_len        = len(self.supp)
                self.lead_data    = True
                self.max_pd       = max(self.depths[-1,:])
            else:
                print("Files are not correct\ncheck files and re-run")
                sys.exit(1)
            self.act[:,0]     =   self.act[:,0] * self.density
            self.act[:,1]     =   self.act[:,1] * self.density
            self.act[:,1]     =   .5*(self.act[:,1]**(-2.))
            self.supp[:,1]    =   .5*(self.supp[:,1]**(-2.))
            print('The 210Pb data whih are loaded are\n{}'.format(self.act))
            print('The 210Pb supported data whih are loaded are\n{}'.format(self.supp))
        else:
            print('There is no 210Pb data')
            self.max_pd          = 0
            self.lead_data       = False
        #load 14C data
        if os.path.isfile(self.hfol + self.dirt + '/' + self.Core + '/' + self.Core + '-C.csv') :
                data             =   genfromtxt(self.hfol + self.dirt + '/' + self.Core + '/' + self.Core + '-C.csv', delimiter=',')
                data             =   data[1:,1:]
                self.dates       =   data[data[:,-1] == 0,:]    # calendar dates
                self.data        =   data[data[:,-1] != 0,:]    # 14C dates
                # checks if there is calendar dates
                if len(self.dates[:,1]) == 0 :
                    self.dates_data = False
                    self.max_date   =   0
                    self.min_date   =   1000000000000000.
                else:
                    self.dates_data = True
                    print('The calendar dates which are loaded are\n{}'.format(self.dates))
                    self.max_date   =   max(self.dates[:,2])
                    self.min_date   =   min(self.dates[:,2])
                    self.dates[:,1] =   .5*(self.dates[:,1]**-2)

                # check if there is radiocarbon dates
                if len(self.data[:,1]) == 0 :
                    self.data_data  =   False
                    self.max_data   =   0
                    self.min_data   =   1000000000000000.
                else:
                    self.data_data  =   True
                    self.max_data   =   max(data[:,2])
                    self.min_data   =   min(data[:,2])
                    print(self.min_data)
                    print('The radiocarbon dates which are loaded are\n{}'.format(self.data))
        else:
            print('There is no 14C data or calendar dates')
            self.dates_data = False
            self.data_data  = False
            self.max_data, self.max_date   =   0,0

    def define_model(self):
        if not self.reservoir_eff:
            self.ini_points   = self.ini_points_
            self.support      = self.support_
            self.var_choosing = self.var_choosing_
        else:
            self.ini_points   = self.ini_points_R
            self.support      = self.support_R
            self.var_choosing = self.var_choosing_R
        if self.Ts_mod:
            self.log_dataC    = self.Ux
        else:
            self.log_dataC    = self.UxN
        # which parametrizacion to use
        if  self.tparam:
            self.pend  = self.pendi
            self.alphas  = self.invpendi
        else:
            self.pend  = self.pendi1
            self.alphas  = self.invpendi1
        # creates the obj function
        if array([self.lead_data,self.dates_data,self.data_data]).sum() == 3:   # 210Pb 14C and Cdates
            def obj_R(param):
                self.var_choosing(param)
                objval          = self.ln_prior_lead() + self.ln_like_supp() + self.Ucs() +  self.log_dataC() + self.ln_like_data()
                return objval
        elif array([self.lead_data,self.data_data]).sum() == 2:                 # 210Pb and 14C
            def obj_R(param):
                self.var_choosing(param)
                objval          = self.ln_prior_lead() + self.ln_like_supp() +  self.log_dataC() + self.ln_like_data()
                return objval
        elif array([self.lead_data,self.dates_data]).sum() == 2:                # 210Pb and Cdates
            def obj_R(param):
                self.var_choosing(param)
                objval          = self.ln_prior_lead() + self.Ucs() +  self.ln_like_data()
                return objval
        elif array([self.dates_data,self.data_data]).sum() == 2:                # 14C and Cdates
            def obj_R(param):
                self.var_choosing(param)
                objval          = self.ln_prior_nonlead() + self.Ucs() +  self.log_dataC()
                return objval
        elif self.lead_data:                                                    # 210Pb
            def obj_R(param):
                self.var_choosing(param)
                objval          = self.ln_prior_lead() + self.ln_like_supp() + self.ln_like_data()
                return objval
        elif self.dates_data:                                                   # Cdates
            def obj_R(param):
                self.var_choosing(param)
                objval          = self.ln_prior_nonlead() + self.Ucs()
                return objval
        elif self.data_data:                                                    # 14C
            def obj_R(param):
                self.var_choosing(param)
                objval          = self.ln_prior_nonlead() + self.log_dataC()
                return objval
        self.obj        =   obj_R

    def linreg_supp(self):
        #this part will check for linearity
        rs          =  1
        for a in range(len(self.Data[:,0])-2):
            x       =   array(self.Data[a:,0]).reshape((-1, 1))
            y       =   log(array(self.Data[a:,2]) )
            model   =   LinearRegression().fit(x, y)
            r_sq    =   model.score(x, y)
            if rs>r_sq:
                a_,rs  =   a,r_sq
        return a_

    def ini_points_(self):
        # parameter order th0,ms,w
        x0_1        = uniform.rvs(size=1, loc=1950-self.Sdate-.0001, scale=.0002)
        m_ini_1     = gamma.rvs(size=self.m,a=self.shape_acc,scale=self.scale_acc)#uniform.rvs(size=self.m, loc=0, scale=15)
        w_ini1      = beta.rvs(size=1,a=self.shape1_m,b=self.shape2_m)#uniform.rvs(size=1, loc=.2, scale=.3)
        #parameters order fi and supported
        if self.lead_data:
            fi_ini      = gamma.rvs(size=1,a=self.fi_shape,scale=self.fi_scale)#uniform.rvs(size=1, loc=0, scale=100)
            s_ini       = gamma.rvs(size=self.s_len,a=self.s_shape,scale=self.s_scale)#)uniform.rvs(size=self.s_len, loc=0, scale=5)
            x           = append(append(append(x0_1,append( m_ini_1 , w_ini1 )),fi_ini),s_ini)
            while not self.support(x):
                m_ini_1 = uniform.rvs(size=self.m, loc=0, scale=1)
                x       = append(append(append(x0_1,append( m_ini_1 , w_ini1 )),fi_ini),s_ini)
        else:
            x           = append(x0_1,append( m_ini_1 , w_ini1 ))
            while not self.support(x):
                m_ini_1 = uniform.rvs(size=self.m, loc=0, scale=1)
                x       = append(x0_1,append( m_ini_1 , w_ini1 ))
        return x

    def ini_points_R(self):
        # parameter order th0,ms,w,reservoir effect
        x0_1        = uniform.rvs(size=1, loc=1950-self.Sdate-.0001, scale=.0002)
        m_ini_1     = gamma.rvs(size=self.m,a=self.shape_acc,scale=self.scale_acc)#uniform.rvs(size=self.m, loc=0, scale=15)
        w_ini1      = beta.rvs(size=1,a=self.shape1_m,b=self.shape2_m)#uniform.rvs(size=1, loc=.2, scale=.3)
        r_ini       = uniform.rvs(size=1, loc=self.r_effect_prior, scale=100)
        #parameters order fi and supported
        if self.lead_data:
            fi_ini      = gamma.rvs(size=1,a=self.fi_shape,scale=self.fi_scale)#uniform.rvs(size=1, loc=0, scale=100)
            s_ini       = gamma.rvs(size=self.s_len,a=self.s_shape,scale=self.s_scale)
            x           = append(append(append(append(x0_1,append( m_ini_1 , w_ini1 )),r_ini),fi_ini),s_ini)
            while not self.support(x):
                m_ini_1 = uniform.rvs(size=self.m, loc=0, scale=1)
                x       = append(append(append(append(x0_1,append( m_ini_1 , w_ini1 )),r_ini),fi_ini),s_ini)
        else:
            x       = append(append(x0_1,append( m_ini_1 , w_ini1 )),r_ini)
            while not self.support(x):
                m_ini_1 = uniform.rvs(size=self.m, loc=0, scale=1)
                x       = append(append(x0_1,append( m_ini_1 , w_ini1 )),r_ini)
        return x

    def support_(self,param):
        self.var_choosing(param)
        if self.lead_data:
            tl = log(self.paramPb[0]*self.Al)/self.lam
            tf = self.times(self.depths[-1,1])
            s0 = tf > tl
            s3 = param[0]   <   1950 - self.Sdate - .01
            s4 = param[0]   >   1950 - self.Sdate + .01
        else:
            s0 = False
            s3 = param[0]   <   1950.-int(strftime("%Y"))
            s4 = False
        s1 = array(param[1:]<=0.).sum()          #Check that every parameter except Th0 are below 0
        s2 = param[self.m+1] >=  1.                   #Checks that w is not >=1
               #Checks that th0 is in reasonable limits
        if s0 + s1 + s2 + s3 + s4 == 0:
            return True
        else:
            return False

    def support_R(self,param):
        self.var_choosing(param)
        if self.lead_data:
            tl = log(self.paramPb[0]*self.Al)/self.lam
            tf = self.times(self.depths[-1,1])
            s0 = tf > tl
        else:
            s0 = False
        s1 = array(delete(param, [0,self.m+2])<=0.).sum()          #Check that every self.parameter except Th0 are below 0
        s2 = param[self.m+1] >=  1.                   #Checks that w is not >=1
        s3 = param[0]        <   1950 - self.Sdate - .0001
        s4 = param[0]        >   1950 - self.Sdate + .0001                  #Checks that th0 is in reasonable limits
        if s0 + s1 + s2 + s3 + s4 == 0:
            return True
        else:
            return False

    def times(self,x):
        x   = array(x)
        ms  = self.pend()
        ys  = append(self.param[0],cumsum(ms * self.by ) + self.param[0])
        ages= array([])
        ages= interp(x,self.breaks,ys)
        return ages

    def pendi(self):
        w       = self.param[self.m+1]
        a       = self.param[1:self.m+1]
        ws      = array(( w*triu(self.matrixone, k=0) + tril(self.matrixone,k=0)) - identity(self.m-1)).prod(axis=1)
        asmt    = a[:-1][::-1] * self.matrixone#
        asmt    = asmt[self.rows, self.column_indices] * triu(self.matrixone, k=0)
        return append(a[-1]*ws+(1-w)*array(asmt*ws/w).sum(axis=1),a[-1] )

    def pendi1(self):
        a   = self.param[1:self.m+1]
        return a

    def invpendi(self):
        return self.param[1:self.m+1]

    def invpendi11(self):
        w   = self.param[self.m+1]
        ms  = self.param[1:self.m+1]
        a   = array([])
        for k in nditer(range(len(ms)-1)):
            a = append(a, (ms[k+1]+(w*ms[k]))/(1-w) )
        a = append(a,ms[self.m-1])
        return ms

    def invpendi1(self):
        w   = self.param[self.m+1]
        ms  = self.param[1:self.m+1]
        alf = (ms[:-1]-w*ms[1:])/(1-w)
        alf = append(alf,ms[-1])
        return ms

    def incallookup(self,points):
        mean    =   self.cc_mean(points)
        var     =   self.cc_var(points)
        result  =   array([mean, var])
        return result
    # Prior distirbution
    def ln_prior_nonlead(self):
        # prior for memory
        logw    = log(self.param[self.m+1])
        prior   = self.iby*(1.-self.shape1_m)*logw + (1.-self.shape2_m)*log(1.-exp(self.iby*logw) ) #+ (1.0-self.iby)*logw - self.logby
        # c version of this: rsc*(1.0-a)*logw + (1.0-b)*log(1.0-exp(rsc*logw)) + (1.0-rsc)*logw - logrsc
        # prior for accumulation rate
        alf     = self.alphas()
        prior   = prior + array((1. - self.shape_acc)*log(alf)+(alf/self.scale_acc)).sum()
        # prior for r_effect
        prior   = prior +  (self.r_effect**2.)*self.r_effect_sd
        return prior

    def ln_prior_lead(self):
        # prior for memory
        logw    = log(self.param[self.m+1])
        prior   = self.iby*(1. - self.shape1_m)*logw + (1. - self.shape2_m)*log(1. - exp(self.iby*logw) ) + (1.0 - self.iby)*logw - self.logby
        # prior for alphas
        alf     = self.alphas()
        prior   = prior + array((1.0 - self.shape_acc)*log(alf) + (alf/self.scale_acc)).sum()
        # prior for fi
        prior   = prior + ((1. - self.fi_shape)*log(self.paramPb[0]) + (self.paramPb[0]/self.fi_shape))
        # prior for supported
        prior   = prior + array(((1. - self.s_shape)*log(self.paramPb[1:])+(self.paramPb[1:]/self.s_scale)) ).sum()
        # prior for r_effect
        prior   = prior +  ((self.r_effect-self.r_effect_prior)**2.)*self.r_effect_sd
        return prior
    # Radiocarbon likelihoods
    def Ux(self):
        dat     = self.times(self.data[:, 2])
        inc     = self.incallookup(dat)
        sigm    = inc[1,:]**2 + self.data[:, 1]**2
        u       = array(( (7./2.) * log(4. + ((array(self.data[:, 0]) - self.r_effect - inc[0,])**2.)/(2.*sigm)) + .5 * log(sigm) ) ).sum()
        return u

    def UxN(self):
        dat     = self.times(self.data[:, 2])
        inc     = self.incallookup(dat)
        sigm    = inc[1,:]**2+self.data[:, 1]**2
        u       = array( ( ((self.data[:, 0] - self.r_effect - inc[0,])**2.) / (2.*sigm)) + .5*log(sigm) ).sum() #
        return u
    # calendar dates
    def Ucs(self):
        dat     = self.times(self.dates[:, 2])
        u       = array((((dat- self.dates[:, 0])**2.)*self.dates[:,1]) ).sum() #
        return u
    # 210Pb likelihoods
    def ln_like_data(self):
        Asup    = self.paramPb[1:] * self.density
        tmp2    = self.paramPb[0]/self.lam
        ts      = -self.lam*( self.times(self.depths[:,1]) - self.param[0] )
        ts0     = -self.lam*( self.times(self.depths[:,0]) - self.param[0] )
        A_i     = Asup + ( tmp2 * (exp(ts0) - exp(ts)) )
        loglike = array( self.act[:,1]*((A_i-self.act[:,0])**2.) ).sum()
        return loglike

    def ln_like_T(self):
        Asup    = self.paramPb[1:] * self.density
        tmp2    = self.paramPb[0]/self.lam
        ts      = self.times(self.depths[:,1]) - self.param[0]
        ts0     = self.times(self.depths[:,0]) - self.param[0]
        A_i     = Asup + tmp2 * (exp(-self.lam*ts0) - exp(-self.lam*ts))
        loglike = array(3.5 * log(4. + self.act[:,1] * ((A_i-self.act[:,0])**2.)) ).sum()
        return loglike

    def ln_like_supp(self):
        return array( self.supp[:,1]*( ( self.supp[:,0]-self.paramPb[1:] )**2. ) ).sum()

    # Creates function that sets parameters internally
    def var_choosing_R(self,param):
        self.r_effect   = param[self.m+2]
        self.param      = param[:self.m+2]
        self.paramPb    = param[self.m+3:]

    def var_choosing_(self,Param):
        self.r_effect   = 0.
        self.param      = Param[:self.m+2]
        self.paramPb    = Param[self.m+2:]

    def Calibrate(self,points,dat):
        inc     = self.incallookup(points)
        sigm    = inc[1,:]**2 + dat[1]**2
        u       = exp( -((dat[0]-inc[0,])**2.)/((2.*sigm))  )* (sigm**-2)
        return u

    def PlumPlot(self):
        # Set up the axes with gridspec
        if self.lead_data:
            fig         = figure(figsize=(10, 10))
            grid        = GridSpec(10, 10, hspace=1.1, wspace=1.1)
            Chronology  = fig.add_subplot(grid[ 2: ,: ])
            Energy      = fig.add_subplot(grid[0:2,0:2], yticklabels=[])
            Acrate      = fig.add_subplot(grid[0:2,2:4], yticklabels=[])
            Memory      = fig.add_subplot(grid[0:2,4:6 ], yticklabels=[])
            fi          = fig.add_subplot(grid[0:2,6:8 ], yticklabels=[])
        else:
            fig         = figure(figsize=(6, 6))
            grid        = GridSpec(6, 6, hspace=1.1, wspace=1.1)
            Chronology  = fig.add_subplot(grid[ 2: ,: ])
            Energy      = fig.add_subplot(grid[0:2,0:2], xticklabels=[])
            Acrate      = fig.add_subplot(grid[0:2,2:4], yticklabels=[])
            Memory      = fig.add_subplot(grid[0:2,4: ], yticklabels=[])
        # Generate chronology
        yrs_it = zeros((1,len(self.breaks)))
        for param in self.Output[1:,:-1]:
            self.var_choosing(param)
            ms      = self.pend()
            ys      = self.times(self.breaks)#append(param[0],cumsum(ms * self.by ) + param[0])
            plt     = Chronology.plot(self.breaks,ys, color='black',alpha = .02)
            yrs_it  = r_[yrs_it,[ys] ]
        yrs_it      = sort(yrs_it[1:,:],axis=0)
        self.ages   =   yrs_it
        # mean and interval
        Chronology.plot(self.breaks,mean(yrs_it,axis=0), linestyle='dashed',c='red',alpha = .9)
        Chronology.plot(self.breaks,yrs_it[int(self.intv * self.iterations) ], linestyle='dashed',c='red',alpha = .9)
        Chronology.plot(self.breaks,yrs_it[int((1-self.intv) * self.iterations) ], linestyle='dashed',c='red',alpha = .9)
        # Plot 14C dates
        if self.data_data:
            for k in self.data:
                nn      = 100
                dates   = interp(k[0],self.ic[:,1],self.ic[:,0])
                y       = linspace(dates-250,dates+250,nn)
                yx      = array(self.Calibrate(y,k) )
                yx      = ((yx-yx.min()) / yx.max()) * self.g_thi-.1
                y       = y[logical_and(yx>5e-03,yx<self.g_thi)]
                yx      = yx[logical_and(yx>5e-03,yx<self.g_thi)]
                x       = repeat(k[2],len(yx))
                Chronology.plot(x+yx,y,color='blue',alpha = .8,lw=.65)
                Chronology.plot(x-yx,y,color='blue',alpha = .8,lw=.65)
                Chronology.fill_betweenx(y,x-yx,x+yx, color='blue',alpha = .5)

                if self.reservoir_eff:
                    datesr   = interp(k[0] - mean(self.outreser[1:]),self.ic[:,1],self.ic[:,0])
                    y       = linspace(datesr-250,datesr+250,nn)
                    kr      = array([k[0] - mean(self.outreser[1:]),k[1],k[2],k[3]])
                    yx      = array(self.Calibrate(y,kr) )
                    yx      = ((yx-yx.min()) / yx.max()) * self.g_thi-.1
                    y       = y[logical_and(yx>5e-04,yx<self.g_thi)]
                    yx      = yx[logical_and(yx>5e-04,yx<self.g_thi)]
                    x       = repeat(k[2],len(yx))
                    Chronology.plot(x+yx,y,color='green',alpha = .8,lw=.65)
                    Chronology.plot(x-yx,y,color='green',alpha = .8,lw=.65)
                    Chronology.fill_betweenx(y,x-yx,x+yx, color='green',alpha = .5)
        # Plot Calendar dates
        if self.dates_data:
            for k in self.dates:
                nn      = 1000
                y       = linspace(k[0]-3*k[1],k[0]+3*k[1],nn)
                yx      = exp(-.5*((k[0]-y)/k[1])**2 )
                yx      = ((yx-yx.min()) / yx.max()) * self.g_thi
                x       = repeat(k[2],len(yx))
                Chronology.plot(x+yx,y,color='deepskyblue',alpha = .8,lw=.65)
                Chronology.plot(x-yx,y,color='deepskyblue',alpha = .8,lw=.65)
                Chronology.fill_betweenx(y,x-yx,x+yx, color='deepskyblue',alpha = .5)
                # plot title and limits
        Chronology.set_xlabel('Depth')
        Chronology.set_ylabel('yr BP')
        Chronology.set_ylim([yrs_it.flatten().min()-5,array(yrs_it[int((1-self.intv)*self.iterations)]).max()+30])
        Chronology.set_xlim([-self.by/12,self.breaks.max()+self.by/12])
        #Chronology.set_title(self.Core)
        string_vals = "{}".format(self.Core)
        Chronology.text(.05,.95, string_vals,transform = Chronology.transAxes,size = 20 )

        # Energy Plot
        Energy.plot(-self.Output[1:,-1],c='gray',lw=.5,alpha=.8)
        Energy.set_title('Log of objective',size=11)

        # Memory kernel
        Memory.set_title('Memory',size=11)
        kr_mem      = gaussian_kde(dataset=self.Outputplt[1:,-2])
        x           = linspace(0,1,200)
        pr_mem      = beta.pdf(x, a=self.shape1_m,b=self.shape2_m )
        string_vals = "mean {}\nshape {}\n".format(self.mean_m,self.shape1_m)
        Memory.plot(x, kr_mem.evaluate(x), linestyle='solid', c='gray', lw=1,alpha=.8)
        Memory.plot(x, pr_mem, linestyle='solid', c='blue', lw=1,alpha=.8)
        Memory.text(.55,.75, string_vals,transform = Memory.transAxes, size = 7,color='red')
        Memory.set_xlim([0,1])

        # Acc kernel
        Acrate.set_title('Acc. Rate',size=11)
        kr_ac       = gaussian_kde(dataset=self.Outputplt[1:,1:-2].flatten())
        x           = linspace(min(self.Outputplt[1:,1:-2].flatten()),max(self.Outputplt[1:,1:-2].flatten()),300)
        pr_acc      = gamma.pdf(x, a=self.shape_acc,scale=self.scale_acc )
        string_vals = "mean {}\nshape {}\nm {}".format(self.mean_acc,self.shape_acc,self.m)
        Acrate.plot(x, kr_ac.evaluate(x),linestyle='solid', c='gray', lw=1,alpha=.8)
        Acrate.plot(x, pr_acc,linestyle='solid', c='blue', lw=1,alpha=.8)
        Acrate.set_xlim([min(self.Outputplt[1:,1:-2].flatten()),max(self.Outputplt[1:,1:-2].flatten())])
        Acrate.text(.55,.75, string_vals,transform = Acrate.transAxes, size = 7,color='red' )

        #plum variables
        if self.lead_data:
            # fi
            fi.set_title('210Pb Infux',size=11)
            kr_fi       = gaussian_kde(dataset=self.outplum[1:,0].flatten())
            x           = linspace(min(self.outplum[1:,0].flatten()),max(self.outplum[1:,0].flatten()),300)
            pr_fi       = gamma.pdf(x, a=self.fi_shape,scale=self.fi_scale )
            string_vals = "mean {}\nshape {}\n".format(self.fi_mean,self.fi_shape)
            fi.plot(x, kr_fi.evaluate(x),linestyle='solid', c='gray', lw=1,alpha=.8)
            fi.plot(x, pr_fi,linestyle='solid', c='blue', lw=1,alpha=.8)
            fi.set_xlim([min(self.outplum[1:,0].flatten()),max(self.outplum[1:,0].flatten())])
            fi.text(.5,.75, string_vals,transform = fi.transAxes, size = 7,color='red')
            # Supp

            x           = linspace(min(self.outplum[1:,1:-1].flatten()),max(self.outplum[1:,1:-1].flatten()),300)
            pr_supp     = gamma.pdf(x, a=self.s_shape,scale=self.s_scale )
            string_vals = "mean {}\nshape {}\n".format(self.s_mean,self.s_shape)

            if self.s_len == 1:
                supp        = fig.add_subplot(grid[0:2,8: ], yticklabels=[])
                kr_supp     = gaussian_kde(dataset=self.outplum[1:,1:-1].flatten())
                supp.plot(x, kr_supp.evaluate(x),linestyle='solid', c='gray', lw=1,alpha=.8)
                supp.plot(x, pr_supp,linestyle='solid', c='blue', lw=1,alpha=.8)
                supp.set_xlim([min(self.outplum[1:,1:-1].flatten()),max(self.outplum[1:,1:-1].flatten())])
                #plotdata
                pltdata     = Chronology.twinx()
                for k in range(len(self.Data[:,1])):
                    y   =   array([self.Data[k,2]-2*self.Data[k,3],self.Data[k,2]-2*self.Data[k,3],self.Data[k,2]+2*self.Data[k,3],self.Data[k,2]+2*self.Data[k,3],self.Data[k,2]-2*self.Data[k,3]]).flatten()
                    x   =   array([self.Data[k,0],self.Data[k,0]-self.Data[k,4],self.Data[k,0]-self.Data[k,4],self.Data[k,0],self.Data[k,0]])
                    if k >= len(self.Data[:,0])-len(self.supp[:,0]):
                        color   =   'red'
                    else:
                        color   =   'blue'
                    pltdata.plot(x,y , c=color,alpha=.3)
            else:
                supp        = fig.add_subplot(grid[0:2,8: ])
                supp.boxplot(self.outplum[1:,1:-1],labels=self.depths[:,1],showfliers=False,vert=False )
                supp.plot(x,pr_supp/max(pr_supp),linestyle='solid', c='blue', lw=1,alpha=.5)
                pltdata     = Chronology.twinx()
                for k in range(len(self.Data[:,1])):
                    y   =   array([self.Data[k,2]-2*self.Data[k,3],self.Data[k,2]-2*self.Data[k,3],self.Data[k,2]+2*self.Data[k,3],self.Data[k,2]+2*self.Data[k,3],self.Data[k,2]-2*self.Data[k,3]]).flatten()
                    x   =   array([self.depths[k,0],self.depths[k,1],self.depths[k,1],self.depths[k,0],self.depths[k,0]])
                    pltdata.plot(x,y , c='blue',alpha=.3)
                    y   =   array([self.Data[k,5]-2*self.Data[k,6],self.Data[k,5]-2*self.Data[k,6],self.Data[k,5]+2*self.Data[k,6],self.Data[k,5]+2*self.Data[k,6],self.Data[k,5]-2*self.Data[k,6]]).flatten()
                    x   =   array([self.depths[k,0],self.depths[k,1],self.depths[k,1],self.depths[k,0],self.depths[k,0]])
                    pltdata.plot(x,y , c='red',alpha=.3)
                    #supp.tick_params(axis="x", labelsize=5)
            pltdata.set_ylabel('210Pb data', color='b')
            supp.set_title('Supp 210Pb',size=11)
            supp.text(.55,.75, string_vals,transform = supp.transAxes, size =7 ,color='red')

        if self.reservoir_eff:
            reserv  = fig.add_subplot(grid[ 3:5 ,3:5 ],yticklabels=[])
            kr_r    = gaussian_kde(dataset=self.outreser[1:])
            x       = linspace(min(self.outreser[1:]),max(self.outreser[1:]),300)
            reserv.set_title('Reservoir effect',size=11)
            reserv.plot(x, kr_r.evaluate(x),linestyle='solid', c='gray', lw=1,alpha=.8)
        #saving plot
        savefig(self.hfol + self.dirt + '/' + self.Core + '/' + self.pdfname,bbox_inches = 'tight')
        if self.showchrono:
            show(fig)
        #plot the reservoir effect if chossen
    def generate_age_file(self):
        depths      = array(arange(self.breaks[0],self.breaks[-1],self.d_by) )
        low         = interp(depths,self.breaks,self.ages[int(self.intv*self.iterations)])
        hig         = interp(depths,self.breaks,self.ages[int((1-self.intv)*self.iterations)])
        mean1       = interp(depths,self.breaks,mean(self.ages,axis=0))
        median1     = interp(depths,self.breaks,median(self.ages,axis=0))
        ages        = array([depths,low ,hig,mean1,median1])
        savetxt(self.hfol + self.dirt + '/' + self.Core + '/' + "ages_{}_{}_{}.txt".format(self.Core, self.m,self.d_by), ages.T, delimiter=',',fmt='%1.3f',header="depth,min,max,mean,median")
        simu        = row_stack((self.breaks,self.ages))
        savetxt(self.hfol + self.dirt + '/' + self.Core + '/' + "Simulaltions_{}_{}_{}.txt".format(self.Core, self.m,self.d_by), simu.T, delimiter=',',fmt='%1.3f')




    def runPlum(self):
        # set seed for replication
        seed(self.seeds)
        # set initial points
        if self.reservoir_eff:
            x,xp           = self.ini_points_R() , self.ini_points_R()
        else:
            x,xp           = self.ini_points() , self.ini_points()
        print('Total iterations are {}'.format(self.thi*self.iterations + self.burnin))
        U, Up          = self.obj(x), self.obj(xp)
        leadchrono     = pytwalk.pytwalk(n=len(x), U=self.obj, Supp=self.support,ww=[ 0.0, 0.4918, 0.4918, 0.0082+0.082, 0.0])   #
        i, k, k0, n    = 0, 0, 0, len(x)
        Output         = zeros((self.iterations+1, n+1))
        Output[0, 0:n] = x.copy()
        Output[0, n]   = U
        por, por2      = int(self.iterations/10.), int(self.burnin/5.)
        while i < self.iterations:
            onemove = leadchrono.onemove(x, U, xp, Up)
            k += 1
            if (all([k < self.burnin, k % por2 == 0])):
                print("Burn-in {}".format(int(100*(k+.0)/self.burnin)) )
            if (uniform.rvs() < onemove[3]):
                x, xp, ke, A, U, Up = onemove
                k0 += 1
                if all([k % self.thi == 0, k > int(self.burnin)]):
                    Output[i+1, 0:n] = x.copy()
                    Output[i+1, n] = U
                    if any([i % por == 0, i == 0]):
                        print('{}%'.format(int(100*(i+.0)/self.iterations)) )
                    i += 1
            else:
                if all([k % self.thi == 0, k > int(self.burnin)]):
                    Output[i+1, 0:n] = x.copy()
                    Output[i+1, n] = U
                    if any([i % por == 0, i == 0]):
                        print('{}%'.format(int(100*(i+.0)/self.iterations)) )
                    i += 1
        print("Acceptance rate")
        print(k0/i + .0)
        Core_name   =   "{}_{}".format(self.Core, self.m)
        savetxt(self.hfol + self.dirt + '/' + self.Core + '/' + Core_name + ".out", Output[:,append(range(self.m+2),-1)], delimiter=',',fmt='%1.3f')
        self.Output         = Output
        self.Outputplt      = Output[:,append(range(self.m+2),-1)]
        self.accpt_rt       = k/k0
        if self.reservoir_eff:
            self.outreser   = Output[:,self.m+2]
            savetxt(self.hfol + self.dirt + '/' + self.Core + '/' + Core_name + "_Reservour.out", self.outreser, delimiter=',',fmt='%1.3f')
            if self.lead_data:
                self.outplum    = Output[:,self.m+3:]
                savetxt(self.hfol + self.dirt + '/' + self.Core + '/' + Core_name + "_Plum.out", self.outplum[:,:-1], delimiter=',',fmt='%1.3f')
        else:
            if self.lead_data:
                self.outplum    = Output[:,self.m+2:]
                savetxt(self.hfol + self.dirt + '/' + self.Core + '/' + Core_name + "_Plum.out", self.outplum[:,:-1], delimiter=',',fmt='%1.3f')
        #generate and save plot
        self.PlumPlot()
        #Save interval file
        self.generate_age_file()


#import importlibimportlib.reload(PyPlum);jc = PyPlum.Plum("simu",r_effect_prior=0.,r_effect_psd=500.,reservoir_eff=True,iterations=100,burnin=10,thi=2);jc.runPlum()
