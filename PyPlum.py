###################################################
###  This is a python implementation of Plum    ###
###  By Marco A. Aquino-Lopez                   ###
###  cite:  Aquino-Lopez, et al. (2018)         ###
###  DOI: 10.1007/s13253-018-0328-7             ###
###################################################

import cProfile
import sys
from tqdm import tqdm
try:
    from numpy import flip,where,quantile,seterr,ogrid, newaxis, arange, triu, ones, tril, identity,median, delete, logical_and,nditer,r_, sort, append, concatenate, repeat, linspace, interp, genfromtxt, array, exp, log, sum,  savetxt, mean, matrix, sqrt, zeros, cumsum, row_stack,vstack,hstack
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
    from matplotlib.pyplot import rc, Line2D, GridSpec, plot, close, show, savefig, hist, xlabel, ylabel, title, axis, subplot, figure, setp, fill_betweenx,xlim,ylim
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
    def __init__(self,Core='HP1C',dirt="/Documents/PyPlum",Dircc="/Documents/PyPlum/Calibration Curves/",
                thick=1.,n_supp=True,  #model parameters
                mean_m=.5,shape_m=5.,mean_acc=10,shape_acc=1.5,cc=True,ccpb="NONE", # bacon paramters
                fi_mean=50., fi_shape=2,s_mean=10,s_shape=5.,Al=.1,s_model = 2, # plum parameters
                reservoir_eff=False,r_effect_prior=0.,r_effect_psd=500., # reservour plum parameters
                iterations=2500,burnin=4000,thi=25, # twalk parameters
                intv=.95,showchrono=False,  # plot parameters
                Ts_mod=True, # use T distribution or normal distribution 
                tparam=False, # tparams referes to which parammetrization to use # True: simulate alphas, False: simulates ms
                g_thi=2,Sdate=True,seed=True,d_by=1.,plotresults=True):
        self.hfol	        =   os.path.expanduser("~")
        # Define seeds
        if seed:
            self.seeds      =   randint(2,9000)
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
        self.fi_mean        =   fi_mean
        self.fi_scale       =   fi_shape/fi_mean # we invert for quicker calculation
        self.s_mean         =   s_mean
        self.s_scale        =   s_shape/s_mean # we invert for quicker calculation
        self.shape2_m       =   (self.shape1_m*(1-mean_m))/mean_m
        # invertimos aqui para hacer mas rapido el codigo
        self.scale_acc      =   shape_acc/mean_acc # Scale of accumulation rate, 
        self.plotresults    =   plotresults
        # MCMC parameters
        self.iterations     =   int(iterations)
        # Extra parameters
        self.tparam         =   tparam     # True: simulate alphas, False: simulates ms
        self.intv           =   intv       # Intervals
        self.s_model        =   s_model 
        # Load data
        print("Working in \n" + self.hfol + self.dirt + '/' + self.Core + '/'+ " \nfolder") 
        self.Sdate          =   Sdate
        self.load_data()
        # Load Calibration Curve
        if self.data_data:
            self.cc             =   cc
            if ccpb == "NONE":
                ccpb = 1     
            self.ccpb           =   ccpb
            self.Dircc          =   Dircc
            self.load_calcurve()
        # setting bacon sections
        self.by             =   thick         # Thickness of bacon seccion
        self.def_breaks()
        self.m              =   len(self.breaks ) - 1 # Number of sections
        # setting and other variables
        self.intv           =   (1-intv)/2. # set the level of probability in the credible interval     
        self.thi            =   int(self.m) * thi   # set the thinning 
        self.burnin         =   int(burnin * self.m ) # set the burn-in 
        #filename and constants
        self.lam            =   0.03114
        self.one_over_lam       =   1./self.lam
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
        # tmp_matrix          =   ones([self.m,self.m])
        self.rows, column_indices = ogrid[:self.matrixone.shape[0], :self.matrixone.shape[1]]
        r                   =  -array(range(self.m-1))[::-1] - 1
        r[r < 0]            += self.matrixone.shape[1]
        self.column_indices = column_indices - r[:, newaxis]
        self.Al             = 1/(self.lam*Al)
        settings            = array([self.d_by,self.shape1_m,self.mean_m,self.shape_acc,self.mean_acc,self.fi_shape,1/self.s_shape,self.fi_scale,self.fi_mean,1/self.s_scale,self.s_mean,self.shape2_m,1/self.scale_acc,self.by,self.m,self.g_thi])
        Core_name   =   "{}_{}".format(self.Core, self.m)
        #self.Output    =   genfromtxt(self.hfol + self.dirt + '/' + self.Core + '/' + Core_name + ".out", delimiter=',')  
        if os.path.isfile(self.hfol + self.dirt + '/' + self.Core + '/' + Core_name + ".out") :
            print(" An existing run was found and will be loaded.\n")
            print(" If you want to rerun and save a previous run, move the files to a new location. \n")
            self.load_old_run()
        else:
            savetxt(self.hfol + self.dirt + '/' + self.Core + '/' + "{}_settings.txt".format(self.Core), settings, delimiter=',',fmt='%1.3f')
            with open(self.hfol + self.dirt + '/' + self.Core + '/' + "{}_settings.txt".format(self.Core), 'a+') as file:
                file.write(str(self.Ts_mod));file.write("\n")
                file.write(str(self.reservoir_eff));file.write("\n")
                file.write(str(self.tparam));file.write("\n")
                for i in self.breaks:
                    file.write(str(i));file.write(",")



    def def_breaks(self):
        if self.lead_data:
            self.breaks         =   array(arange(0,max(self.max_pd,self.max_data,self.max_date)+2*self.by,self.by))
        else:
            self.breaks         =   array(arange(min(self.min_data,self.min_date),max(self.max_data,self.max_date)+2*self.by,self.by))


#################################################################
#################################################################
################################################################# 
    #def load_multicalcurve(self):
    def load_calcurve(self):
        self.data[where(self.data[:, 0] < 0) , 3] = self.ccpb + 3
        print("Loading calibration curves")
        cc = "IntCal20.14C"
        intcal1              =   genfromtxt(self.hfol + self.Dircc + cc, delimiter=',')
        self.ic1             =   intcal1
        self.cc_mean1        =   interp1d(self.ic1[:,0],self.ic1[:,1], fill_value="extrapolate")
        self.cc_var1         =   interp1d(self.ic1[:,0],self.ic1[:,2], fill_value="extrapolate")
        cc = 'Marine20.14C'
        intcal2              =   genfromtxt(self.hfol + self.Dircc + cc, delimiter=',')
        self.ic2             =   intcal2
        self.cc_mean2        =   interp1d(self.ic2[:,0],self.ic2[:,1], fill_value="extrapolate")
        self.cc_var2         =   interp1d(self.ic2[:,0],self.ic2[:,2], fill_value="extrapolate")
        cc = 'SHCal20.14C'
        intcal3              =   genfromtxt(self.hfol + self.Dircc + cc, delimiter=',')
        self.ic3             =   intcal3
        self.cc_mean3        =   interp1d(self.ic3[:,0],self.ic3[:,1], fill_value="extrapolate")
        self.cc_var3         =   interp1d(self.ic3[:,0],self.ic3[:,2], fill_value="extrapolate")
        #here we load postbombs
        ccpb = "postbomb_NH1_monthly.14C"
        ccpb1                  =   genfromtxt(self.hfol + self.Dircc + ccpb, delimiter=',')
        self.pbcc1             =   ccpb1#flip(ccpb1,axis=0)
        self.pbcc_mean1        =   interp1d(self.pbcc1[:,0],self.pbcc1[:,1], fill_value="extrapolate")
        self.pbcc_var1         =   interp1d(self.pbcc1[:,0],self.pbcc1[:,2], fill_value="extrapolate")
        ccpb = "postbomb_NH2_monthly.14C"
        ccpb2                  =   genfromtxt(self.hfol + self.Dircc + ccpb, delimiter=',')
        self.pbcc2             =   ccpb2#flip(ccpb2,axis=0)
        self.pbcc_mean2        =   interp1d(self.pbcc2[:,0],self.pbcc2[:,1], fill_value="extrapolate")
        self.pbcc_var2         =   interp1d(self.pbcc2[:,0],self.pbcc2[:,2], fill_value="extrapolate")
        ccpb = "postbomb_NH3_monthly.14C"
        ccpb3                  =   genfromtxt(self.hfol + self.Dircc + ccpb, delimiter=',')
        self.pbcc3             =   ccpb3#flip(ccpb3,axis=0)
        self.pbcc_mean3        =   interp1d(self.pbcc3[:,0],self.pbcc3[:,1], fill_value="extrapolate")
        self.pbcc_var3         =   interp1d(self.pbcc3[:,0],self.pbcc3[:,2], fill_value="extrapolate")
        ccpb = "postbomb_SH3_monthly.14C"
        ccpb4                  =   genfromtxt(self.hfol + self.Dircc + ccpb, delimiter=',')
        self.pbcc4             =   ccpb4#flip(ccpb4,axis=0)
        self.pbcc_mean4        =   interp1d(self.pbcc4[:,0],self.pbcc4[:,1], fill_value="extrapolate")
        self.pbcc_var4         =   interp1d(self.pbcc4[:,0],self.pbcc4[:,2], fill_value="extrapolate")
        ccpb = "postbomb_SH1-2_monthly.14C"
        ccpb5                  =   genfromtxt(self.hfol + self.Dircc + ccpb, delimiter=',')
        self.pbcc5             =   ccpb5#flip(ccpb5,axis=0)
        self.pbcc_mean5        =   interp1d(self.pbcc5[:,0],self.pbcc5[:,1], fill_value="extrapolate")
        self.pbcc_var5         =   interp1d(self.pbcc5[:,0],self.pbcc5[:,2], fill_value="extrapolate")

        self.ic              =  [intcal1,intcal2,intcal3,ccpb1,ccpb2,ccpb3,ccpb4,ccpb5]
        self.cc_mean         =  [self.cc_mean1,self.cc_mean2,self.cc_mean3,self.pbcc_mean1,self.pbcc_mean2,self.pbcc_mean3,self.pbcc_mean4,self.pbcc_mean5]
        self.cc_var          =  [self.cc_var1,self.cc_var2,self.cc_var3,self.pbcc_var1,self.pbcc_var2,self.pbcc_var3,self.pbcc_var4,self.pbcc_var5]

#################################################################
#################################################################
#################################################################            
        

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
                if self.s_model ==  1: 
                    self.s_len    = 1
                else:
                    self.s_len        = len(self.supp)
                self.lead_data    = True
                self.max_pd       = max(self.depths[-1,:])
            else:
                print("Files are not correct\ncheck files in \n" + self.hfol + self.dirt + '/' + self.Core + '/' + self.Core + '.csv\n' + " and try again") ; print(self.Core)
                sys.exit(1)
            self.act[:,0]     =   self.act[:,0] * self.density
            self.act[:,1]     =   self.act[:,1] * self.density
            self.act[:,1]     =   .5*(self.act[:,1]**(-2.))
            self.supp[:,1]    =   .5*(self.supp[:,1]**(-2.))
            print('210Pb data found and loaded\n{}'.format(self.act))
            print('210Pb supported data found and loaded\n{}'.format(self.supp))
        else:
            print('There is no 210Pb data.\nRuning model without 210Pb data')
            self.max_pd          = 0
            self.lead_data       = False
        #load 14C data
        if os.path.isfile(self.hfol + self.dirt + '/' + self.Core + '/' + self.Core + '-C.csv') :
                data             =   genfromtxt(self.hfol + self.dirt + '/' + self.Core + '/' + self.Core + '-C.csv', delimiter=',')
                data             =   data[1:,1:]
                self.dates       =   data[data[:,-1] == 0,:]    # calendar dates
                self.data        =   data[data[:,-1] != 0,:]    # 14C dates
                self.data        =   data[data[:,-1] != 0,:]    # 14C dates
                # checks if there is calendar dates
                if len(self.dates[:,1]) == 0 :
                    self.dates_data = False
                    self.max_date   =   0
                    self.min_date   =   1e+16
                else:
                    self.dates_data = True
                    print('Calendar dates found and loaded\n{}'.format(self.dates))
                    self.max_date   =   max(self.dates[:,2])
                    self.min_date   =   min(self.dates[:,2])
                    self.dates[:,1] =   .5*(self.dates[:,1]**-2)

                # check if there is radiocarbon dates
                if len(self.data[:,1]) == 0 :
                    self.data_data  =   False
                    self.max_data   =   0
                    self.min_data   =   1e+16
                else:
                    self.data_data  =   True
                    self.max_data   =   max(data[:,2])
                    self.min_data   =   min(data[:,2])
                    print(self.min_data)
                    print('Radiocarbon dates found and loaded. \n{}'.format(self.data))
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
        ms_ini_1     = gamma.rvs(size=self.m,a=self.shape_acc,scale=1./self.scale_acc)
        if not(self.tparam):
            w_ini1      = uniform.rvs(size=1,loc=0,scale=min(ms_ini_1[:-1] / ms_ini_1[1:]))
        else:
            w_ini1      = beta.rvs(size=1,a=self.shape1_m,b=self.shape2_m) 
        #parameters order fi and supported
        if self.lead_data:
            fi_ini      = gamma.rvs(size=1,a=self.fi_shape,scale=1/self.fi_scale)#uniform.rvs(size=1, loc=0, scale=100)
            s_ini       = gamma.rvs(size=self.s_len,a=self.s_shape,scale=1/self.s_scale)#)uniform.rvs(size=self.s_len, loc=0, scale=5)
            x           = append(append(append(x0_1,append( ms_ini_1 , w_ini1 )),fi_ini),s_ini)
            while not self.support(x):
                m_ini_1 = uniform.rvs(size=self.m, loc=0, scale=1)
                if not(self.tparam):
                    w_ini1      = uniform.rvs(size=1,loc=0,scale=min(ms_ini_1[:-1] / ms_ini_1[1:]))
                else:
                    w_ini1      = beta.rvs(size=1,a=self.shape1_m,b=self.shape2_m) 
                x       = append(append(append(x0_1,append( m_ini_1 , w_ini1 )),fi_ini),s_ini)     
        else:
            x           = append(x0_1,append( m_ini_1 , w_ini1 ))
            while not self.support(x):
                m_ini_1 = uniform.rvs(size=self.m, loc=0, scale=1)
                if not(self.tparam):
                    w_ini1      = uniform.rvs(size=1,loc=0,scale=min(ms_ini_1[:-1] / ms_ini_1[1:]))
                else:
                    w_ini1      = beta.rvs(size=1,a=self.shape1_m,b=self.shape2_m) 
                x       = append(x0_1,append( m_ini_1 , w_ini1 ))
        return x

    def ini_points_R(self):
        # parameter order th0,ms,w,reservoir effect
        x0_1        = uniform.rvs(size=1, loc=1950-self.Sdate-.0001, scale=.0002)
        m_ini_1     = gamma.rvs(size=self.m,a=self.shape_acc,scale=1./self.scale_acc)  #uniform.rvs(size=self.m, loc=0, scale=15)
        if not(self.tparam):
            w_ini1      = uniform.rvs(size=1,loc=0,scale=min(m_ini_1[:-1] / m_ini_1[1:]))
        else:
            w_ini1      = beta.rvs(size=1,a=self.shape1_m,b=self.shape2_m) 
        r_ini       = uniform.rvs(size=1, loc=self.r_effect_prior, scale=100)
        #parameters order fi and supported
        if self.lead_data:
            fi_ini      = gamma.rvs(size=1,a=self.fi_shape,scale=1/self.fi_scale)#uniform.rvs(size=1, loc=0, scale=100)
            s_ini       = gamma.rvs(size=self.s_len,a=self.s_shape,scale=1/self.s_scale)
            x           = append(append(append(append(x0_1,append( m_ini_1 , w_ini1 )),r_ini),fi_ini),s_ini)
            while not self.support(x):
                m_ini_1 = gamma.rvs(size=self.m,a=self.shape_acc,scale=1./self.scale_acc) 
                x       = append(append(append(append(x0_1,append( m_ini_1 , w_ini1 )),r_ini),fi_ini),s_ini)
        else:
            x       = append(append(x0_1,append( m_ini_1 , w_ini1 )),r_ini)
            while not self.support(x):
                m_ini_1 = gamma.rvs(size=self.m,a=self.shape_acc,scale=1./self.scale_acc) 
                x       = append(append(x0_1,append( m_ini_1 , w_ini1 )),r_ini)
        return x

    def support_(self,param):
        self.var_choosing(param)
        if self.lead_data:
            tl = log(self.paramPb[0]*self.Al) * self.one_over_lam  
            tf = self.times(self.depths[-1,1])
            s0 = tf > tl
            #Checks that th0 is in reasonable limits
            s3 = param[0]   <   1950 - self.Sdate - .01
            s4 = param[0]   >   1950 - self.Sdate + .01
        else:
            s0 = False
            s3 = param[0]   <   1950.-int(strftime("%Y"))
            s4 = False
        s1 = array(param[1:]<=0.).sum()          #Check that every parameter except Th0 are below 0
        s2 = param[self.m+1] >=  1.                   #Checks that w is not >=1         
        s5 = any( self.alphas() < 0 )
        if s0 + s1 + s2 + s3 + s4 +s5== 0:
            return True
        else:
            return False

    def support_R(self,param):
        self.var_choosing(param)
        if self.lead_data:
            tl = log(self.paramPb[0]*self.Al)*self.one_over_lam  
            tf = self.times(self.depths[-1,1])
            s0 = tf > tl
            s3 = param[0]        <   1950 - self.Sdate - .0001
            s4 = param[0]        >   1950 - self.Sdate + .0001                  #Checks that th0 is in reasonable limits
        else:
            s0 = False            
            s3 = param[0]   <   1950.-int(strftime("%Y"))
            s4 = False
        s1 = array(delete(param, [0,self.m+2])<=0.).sum()          #Check that every self.parameter except Th0 are below 0
        s2 = param[self.m+1] >=  1.                   #Checks that w is not >=1
        s5 = any( self.alphas() < 0 )
        if s0 + s1 + s2 + s3 + s4 + s5 == 0:
            return True
        else:
            return False

    def times(self,x):
        x   = array(x)
        ms  = self.pend()
        ys  = append(self.param[0], self.param[0] + cumsum(ms * self.by )  )
        # ages= array([])
        ages= interp(x,self.breaks,ys)
        return ages

    def pendi(self):
        w       = self.param[self.m+1]
        # orginal order
        a       = self.param[1:self.m+1]
        # change for right other?
        # a = self.param[1:self.m+1][::-1]
        ws      = array(( w*triu(self.matrixone, k=0) + tril(self.matrixone,k=0)) - identity(self.m-1)).prod(axis=1)
        asmt    = a[:-1][::-1] * self.matrixone
        asmt    = asmt[self.rows, self.column_indices] * triu(self.matrixone, k=0)
        # orginal order
        # return append(a[-1]*ws+(1-w)*array(asmt*ws/w).sum(axis=1),a[-1] )
        return append(a[0] * ws + (1 - w) * array(asmt * ws / w).sum(axis=1), a[0])

    def pendi1(self):
        a   = self.param[1:self.m+1]
        return a

    def invpendi(self):
        return self.param[1:self.m+1]

    # def invpendi11(self):
    #     w   = self.param[self.m+1]
    #     ms  = self.param[1:self.m+1]
    #     a   = array([])
    #     for k in nditer(range(len(ms)-1)):
    #         a = append(a, (ms[k+1]+(w*ms[k]))/(1-w) )
    #     a = append(a,ms[self.m-1])
    #     return ms

    def invpendi1(self):
        w   = self.param[self.m+1]
        ms  = self.param[1:self.m+1]
        alf = ( ms[:-1] - w * ms[1:] )/(1-w)
        return append(alf,ms[-1])
    
    # functions for calculating the calibrated ages

    def incallookup(self,points,calicurv):
        if len(calicurv) ==1 :            
            calicurv = array([calicurv])
            if points.size !=1:
                calicurv = repeat(calicurv,points.size)  #AQUI ESTOY AQUI
        else:
            calicurv = array(calicurv)
        mean=array([])
        var=array([])
        for k in range(calicurv.size):
            mean    =   append(mean,self.cc_mean[int(calicurv[k]-1)](points[k]))
            var     =   append(var,self.cc_var[int(calicurv[k]-1)](points[k])  )
        result  =   array([mean, var])
        return result
    
    # Prior distirbution
    def ln_prior_nonlead(self):
        # prior for memory
        logw    = log(self.param[self.m+1])
        prior   = self.iby*(1.-self.shape1_m)*logw + (1.-self.shape2_m)*log(1.-exp(self.iby*logw) ) + (1.0-self.iby)*logw - self.logby
        # c version of this: rsc*(1.0-a)*logw + (1.0-b)*log(1.0-exp(rsc*logw)) + (1.0-rsc)*logw - logrsc
        # prior for accumulation rate
        alf     = self.alphas()
        prior   = prior + array((1. - self.shape_acc)*log(alf)+(alf * self.scale_acc)).sum()
        # c version of this: 1.0-alpha[i])*log(al) + beta[i]*al;
        # prior for r_effect
        prior   = prior +  (self.r_effect**2.)*self.r_effect_sd
        return prior

    def ln_prior_lead(self):
        # prior for memory
        logw    = log(self.param[self.m+1])
        prior   = self.iby * (1. - self.shape1_m)*logw + (1. - self.shape2_m)*log(1. - exp(self.iby*logw) ) + (1.0 - self.iby)*logw - self.logby
        # prior for alphas
        alf     = self.alphas()
        prior   = prior + array( (1.0 - self.shape_acc)*log(alf) + (alf*self.scale_acc) ).sum()
        # prior for fi
        prior   = prior + ((1. - self.fi_shape)*log(self.paramPb[0]) + (self.paramPb[0]*self.fi_scale))
        # prior for supported
        prior   = prior + array(((1. - self.s_shape)*log(self.paramPb[1:])+(self.paramPb[1:]*self.s_scale)) ).sum()
        # prior for r_effect
        prior   = prior +  ((self.r_effect-self.r_effect_prior)**2.)*self.r_effect_sd
        return prior

    #   Radiocarbon likelihoods, 
    #   check calibration curve
    
    def Ux(self): # likelihood using T student 
        dat     = self.times(self.data[:, 2])
        inc     = self.incallookup(dat,self.data[:, 3])
        sigm    = inc[1,:]**2 + self.data[:, 1]**2
        u       = array(( (7./2.) * log(4. + ((array(self.data[:, 0]) - self.r_effect - inc[0,])**2.)/(2.*sigm)) + .5 * log(sigm) ) ).sum()
        return u

    def UxN(self): #likelihood using normal distribution
        dat     = self.times(self.data[:, 2])
        inc     = self.incallookup(dat,self.data[:, 3])
        sigm    = inc[1,:]**2+self.data[:, 1]**2
        u       = array( ( ((self.data[:, 0] - self.r_effect - inc[0,])**2.) / (2.*sigm)) + .5*log(sigm) ).sum() #
        return u
    
    ## Likelihood for calendar dates
    def Ucs(self):
        dat     = self.times(self.dates[:, 2])
        u       = array((((dat- self.dates[:, 0])**2.)*self.dates[:,1]) ).sum() #
        return u
    
    # 210Pb likelihoods
    def ln_like_data(self): #likelihood using normal distribtion
        Asup    = self.paramPb[1:] * self.density
        tmp2    = self.paramPb[0]* self.one_over_lam  
        ts      = -self.lam*( self.times(self.depths[:,1]) - self.param[0] )
        ts0     = -self.lam*( self.times(self.depths[:,0]) - self.param[0] )
        A_i     =  Asup +( tmp2 * (exp(ts0) - exp(ts)) )
        # self.A_test = A_i
        loglike = array( self.act[:,1] * ((  A_i - self.act[:,0] )**2.) ).sum()
        return loglike

    def ln_like_T(self): #likelihood using T student for lead210
        # revisar creo que esa mal
        Asup    = self.paramPb[1:] * self.density
        tmp2    = self.paramPb[0] * self.one_over_lam  
        ts      = self.times(self.depths[:,1]) - self.param[0]
        ts0     = self.times(self.depths[:,0]) - self.param[0]
        A_i     = Asup + tmp2 * (exp(-self.lam*ts0) - exp(-self.lam*ts))
        # self.A_test = A_i
        loglike = array(3.5 * log(4. + self.act[:,1] * ((A_i-self.act[:,0])**2.)) ).sum()
        return loglike

    # Supported activity likelihood
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
        cc = repeat(dat[3],len(points))
        if len(points) > 1:
            indx = where(array(points)<0)[0]
            if len(indx)>0:
                cc[indx] = 3 + self.ccpb
        else:
            if any(points < 0):
                cc[indx] = 3 + self.ccpb
        cc = array(cc)
        inc     = self.incallookup(points,cc)
        sigm    = inc[1,:]**2 + dat[1]**2
        u       = exp( -((dat[0]-inc[0,])**2.)/((2.*sigm))  )* (sigm**-2)
        return u

    def PlumPlot(self,add_name = ''):
        print('Plotting the results')
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

        self.A_i = zeros(len(self.times(self.depths[:,1])))
        for param in self.Output[1:,:-1]:
            self.var_choosing(param)
            # ms      = self.pend()
            ys      = self.times(self.breaks)#append(param[0],cumsum(ms * self.by ) + param[0])
            Chronology.plot(self.breaks,ys, color='black',alpha = .02)
            yrs_it  = r_[yrs_it,[ys] ]
            Asup    = self.paramPb[1:] 
            tmp2    = self.paramPb[0] * self.one_over_lam  
            ts      = -self.lam*( self.times(self.depths[:,1]) - self.param[0] )
            ts0     = -self.lam*( self.times(self.depths[:,0]) - self.param[0] )
            self.A_i   = vstack([self.A_i, Asup + (tmp2 * (exp(ts0) - exp(ts)))/self.density ] )
        self.A_i = self.A_i[1:,:]
            
        yrs_it      = sort(yrs_it[1:,:],axis=0)
        self.ages   =   yrs_it
        # mean and interval
        Chronology.plot(self.breaks,mean(yrs_it,axis=0), linestyle='dashed',c='red',alpha = .9)
        Chronology.plot(self.breaks,yrs_it[int(self.intv * self.iterations) ], linestyle='dashed',c='red',alpha = .9)
        Chronology.plot(self.breaks,yrs_it[int((1-self.intv) * self.iterations) ], linestyle='dashed',c='red',alpha = .9)
        # Plot 14C dates
        if self.data_data:
            for k in self.data:
                nn      = 200
                if k[3] == 1:
                    ic = self.ic1
                if k[3] == 2:
                    ic = self.ic2
                if k[3] == 3:
                    ic = self.ic3
                if k[3] == 4:
                    ic = self.pbcc1
                if k[3] == 5:
                    ic = self.pbcc2
                if k[3] == 6:
                    ic = self.pbcc3
                if k[3] == 7:
                    ic = self.pbcc4
                if k[3] == 8:
                    ic = self.pbcc5
                dates   = interp(k[0],ic[:,1],ic[:,0])
                if k[3] <= 3:
                    y   = linspace(dates-150,dates+150,nn)
                else:
                    y   = linspace(-73,0,1500)
                yx      = array(self.Calibrate(y,k))
                yx      = ((yx-yx.min()) / yx.max()) * self.g_thi-.1
                y       = y[logical_and(yx>5e-03,yx<self.g_thi)]
                yx      = yx[logical_and(yx>5e-03,yx<self.g_thi)]
                x       = repeat(k[2],len(yx))
                Chronology.plot(x+yx,y,color='blue',alpha = .8,lw=.65)
                Chronology.plot(x-yx,y,color='blue',alpha = .8,lw=.65)
                Chronology.fill_betweenx(y,x-yx,x+yx, color='blue',alpha = .5)

                if self.reservoir_eff:
                    datesr  = interp(k[0] - mean(self.outreser[1:]),ic[:,1],ic[:,0])
                    if k[3] <= 3:
                        y   = linspace(datesr-150,datesr+150,nn)
                    else:
                        y   = linspace(-75,0,nn)
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
        Chronology.set_title(self.Core + add_name)
        string_vals = "{}".format(self.Core + add_name)
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
        pr_acc      = gamma.pdf(x, a=self.shape_acc,scale=1./self.scale_acc )
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
            pr_fi       = gamma.pdf(x, a=self.fi_shape,scale=1/self.fi_scale )
            string_vals = "mean {}\nshape {}\n".format(self.fi_mean,self.fi_shape)
            fi.plot(x, kr_fi.evaluate(x),linestyle='solid', c='gray', lw=1,alpha=.8)
            fi.plot(x, pr_fi,linestyle='solid', c='blue', lw=1,alpha=.8)
            fi.set_xlim([min(self.outplum[1:,0].flatten()),max(self.outplum[1:,0].flatten())])
            fi.text(.5,.75, string_vals,transform = fi.transAxes, size = 7,color='red')
            # Supp

            x           = linspace(min(self.outplum[1:,1:-1].flatten()),max(self.outplum[1:,1:-1].flatten()),300)
            pr_supp     = gamma.pdf(x, a=self.s_shape,scale=1/self.s_scale )
            string_vals = "mean {}\nshape {}\n".format(self.s_mean,self.s_shape)

            if self.s_len == 1:
                supp        = fig.add_subplot(grid[0:2,8: ], yticklabels=[])
                kr_supp     = gaussian_kde(dataset=self.outplum[1:,1:-1].flatten())
                supp.plot(x, kr_supp.evaluate(x),linestyle='solid', c='gray', lw=1,alpha=.8)
                supp.plot(x, pr_supp,linestyle='solid', c='blue', lw=1,alpha=.8)
                supp.set_xlim([min(self.outplum[1:,1:-1].flatten()),1.3*max(self.outplum[1:,1:-1].flatten())])
                #plotdata
                pltdata     = Chronology.twinx()
                for k in range(len(self.depths[:,1])):
                    y   =   array([self.Data[k,2]-2*self.Data[k,3],self.Data[k,2]-2*self.Data[k,3],self.Data[k,2]+2*self.Data[k,3],self.Data[k,2]+2*self.Data[k,3],self.Data[k,2]-2*self.Data[k,3]]).flatten()
                    x   =   array([self.Data[k,0],self.Data[k,0]-self.Data[k,4],self.Data[k,0]-self.Data[k,4],self.Data[k,0],self.Data[k,0]])
                    if k >= len(self.Data[:,0])-len(self.supp[:,0]):
                        color   =   'red'
                    else:
                        color   =   'blue'
                    pltdata.plot(x,y , c=color,alpha=.5)
                    for k1 in range(len(self.outplum[1:, 1:-1])):
                        y_value = self.outplum[k1+1, 1] 
                        x_min = self.depths[k, 0]
                        x_max = self.depths[k, 1]
                        pltdata.hlines(y=y_value, xmin=x_min+.09, xmax=x_max-.09, colors='red', alpha=.002)    
                        pltdata.hlines(y=self.A_i[k1,k], xmin=x_min+.09, xmax=x_max-.09, colors='blue', alpha=.02)
            else:
                supp        = fig.add_subplot(grid[0:2,8: ])
                # plot the posterior samples
                supp.boxplot(self.outplum[1:,1:-1],positions=self.depths[:, 1],labels=self.depths[:,1],showfliers=False,vert=False )
                # plot the prior 
                supp.plot(x,pr_supp/max(pr_supp),linestyle='solid', c='blue', lw=1,alpha=.5)
                # plot the data
                supp.errorbar(self.Data[:, 5], self.depths[:,1] , xerr=self.Data[:, 6], fmt='o', color='red', alpha=0.4)
                # change the limit
                supp.set_xlim([min(self.outplum[1:,1:-1].flatten()),1.45*max(self.outplum[1:,1:-1].flatten())])
                # Reduce the size of x-axis labels
                supp.tick_params(axis='y', labelsize=6)
                # this plots the supported data in the main plot
                pltdata     = Chronology.twinx()
                for k in range(len(self.Data[:,1])):
                    y   =   array([self.Data[k,2]-2*self.Data[k,3],self.Data[k,2]-2*self.Data[k,3],self.Data[k,2]+2*self.Data[k,3],self.Data[k,2]+2*self.Data[k,3],self.Data[k,2]-2*self.Data[k,3]]).flatten()
                    x   =   array([self.depths[k,0],self.depths[k,1],self.depths[k,1],self.depths[k,0],self.depths[k,0]])
                    pltdata.plot(x,y , c='blue',alpha=.5)
                    y   =   array([self.Data[k,5]-2*self.Data[k,6],self.Data[k,5]-2*self.Data[k,6],self.Data[k,5]+2*self.Data[k,6],self.Data[k,5]+2*self.Data[k,6],self.Data[k,5]-2*self.Data[k,6]]).flatten()
                    x   =   array([self.depths[k,0],self.depths[k,1],self.depths[k,1],self.depths[k,0],self.depths[k,0]])
                    pltdata.plot(x,y , c='red',alpha=.5)
                    # plot the output
                    y_value = self.outplum[1:, k+1]  # Assuming 1D array; if 2D, you'll have to loop through columns
                    x_min = self.depths[k, 0]
                    x_max = self.depths[k, 1]
                    pltdata.hlines(y=y_value, xmin=x_min+.09, xmax=x_max-.09, colors='red', alpha=.02)    # Choose the color and transparency as you like
                    pltdata.hlines(y=self.A_i[:,k], xmin=x_min+.09, xmax=x_max-.09, colors='blue', alpha=.02)
                
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
        savefig(self.hfol + self.dirt + '/' + self.Core + '/'  + add_name +self.pdfname,bbox_inches = 'tight')
        if self.showchrono:
            show(fig)
        #plot the reservoir effect if chossen
    def generate_age_file(self,add_name= ''):
        depths      = array(arange(self.breaks[0],self.breaks[-1],self.d_by) )
        low         = interp(depths,self.breaks,quantile(self.ages,self.intv,axis=0))
        hig         = interp(depths,self.breaks,quantile(self.ages,1-self.intv,axis=0))
        mean1       = interp(depths,self.breaks,mean(self.ages,axis=0))
        median1     = interp(depths,self.breaks,median(self.ages,axis=0))
        ages        = array([depths,low ,hig,mean1,median1])
        savetxt(self.hfol + self.dirt + '/' + self.Core + '/' + add_name + "ages_{}_{}_{}.txt".format(self.Core, self.m,self.d_by), ages.T, delimiter=',',fmt='%1.3f',header="depth,min,max,mean,median")
        simu        = row_stack((self.breaks,self.ages))
        savetxt(self.hfol + self.dirt + '/' + self.Core + '/' + add_name + "Simulaltions_{}_{}_{}.txt".format(self.Core, self.m,self.d_by), simu.T, delimiter=',',fmt='%1.3f')

    def set_ini(self):
        if self.reservoir_eff:
            x,xp           = self.ini_points_R() , self.ini_points_R()
        else:
            x,xp           = self.ini_points() , self.ini_points()
        return x, xp


    def runPlum(self):
        # set seed for replication
        seed(self.seeds)
        # set initial points
        x, xp = self.set_ini()
        print('Total iterations are {}'.format(self.thi * self.iterations + self.burnin))
        U, Up          = self.obj(x), self.obj(xp)
        twalkrun     = pytwalk.pytwalk(n=len(x), U=self.obj, Supp=self.support)#,ww=[ 0.0, 0.4918, 0.4918, 0.0082+0.082, 0.0])   #
        # total_iterations = self.thi * self.iterations + self.burnin
        # twalkrun.Run(T=total_iterations, x0=x, xp0=xp, thi=self.thi)
        i, k, k0, n    = 0, 0, 0, len(x)
        Output         = zeros((self.iterations+1, n+1))
        Output[0, 0:n] = x.copy()
        Output[0, n]   = U
        # Here we start the while
        pbar = tqdm(total = self.iterations)
        while i < self.iterations:
            onemove = twalkrun.onemove(x, U, xp, Up)
            k += 1
            #if (all([k < self.burnin, k % por2 == 0])):
            #    print("Burn-in {}".format(int(100*(k+.0)/self.burnin)) )
            if (uniform.rvs() < onemove[3]):
                x, xp, ke, A, U, Up = onemove
                k0 += 1
                if all([k % self.thi == 0, k > int(self.burnin)]):
                    Output[i+1, 0:n] = x.copy()
                    Output[i+1, n] = U
                    #if any([i % por == 0, i == 0]):
                    #    print('{}%'.format(int(100*(i+.0)/self.iterations)) )
                    i += 1
                    pbar.update(1)
            else:
                if all([k % self.thi == 0, k > int(self.burnin)]):
                    Output[i+1, 0:n] = x.copy()
                    Output[i+1, n] = U
                    #if any([i % por == 0, i == 0]):
                    #    print('{}%'.format(int(100*(i+.0)/self.iterations)) )
                    i += 1
                    pbar.update(1)
        pbar.close()
        # end of while
        # Output = twalkrun.Output[-self.iterations:, :]
        # save inicial points added 26/04/2023
        # initial_poitns = [twalkrun.x, twalkrun.xp]
        print('\n')
        print('saving Output and state points\n')
        initial_poitns = [x, xp]
        self.x_last = x
        self.xp_last = xp
        # save out file
        Core_name   =   "{}_{}".format(self.Core, self.m)
        savetxt(self.hfol + self.dirt + '/' + self.Core + '/' + Core_name + "initial_poitns.csv", initial_poitns, delimiter=',')
        savetxt(self.hfol + self.dirt + '/' + self.Core + '/' + Core_name + ".out", Output[:,append(range(self.m+2),-1)], delimiter=',',fmt='%1.3f')
        self.Output         = Output
        self.Outputplt      = Output[:,append(range(self.m+2),-1)]
        # self.accpt_rt       = k/k0  # this is no longer needed
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
        if self.plotresults:
            print('Plotting results')
            self.PlumPlot()
            #Save interval file
            self.generate_age_file()

    

    def load_old_run(self):
        # Loader    =   genfromtxt(self.hfol + self.dirt + '/' + self.Core + '/' + "{}_settings.txt".format(self.Core), delimiter=',')    
        # print(Loader)
        Core_name   =   "{}_{}".format(self.Core, self.m)
        self.Output    =   genfromtxt(self.hfol + self.dirt + '/' + self.Core + '/' + Core_name + ".out", delimiter=',')    
        if self.reservoir_eff:
            self.outreser  =   genfromtxt(self.hfol + self.dirt + '/' + self.Core + '/' + Core_name + "_Reservour.out", delimiter=',')
        if self.lead_data:
            self.outplum   =   genfromtxt(self.hfol + self.dirt + '/' + self.Core + '/' + Core_name + "_Plum.out",delimiter=',')
        
        
        
    def SAR_d(self):
        output = self.Output[1:,1:-2]
        if not(self.tparam) :
            depths      = array(arange(self.breaks[0],self.breaks[-1],self.d_by) )
            mean1       = interp(depths,self.breaks[1:],mean(output,axis=0))
            median1     = interp(depths,self.breaks[1:],median(output,axis=0))
            low         = interp(depths,self.breaks[1:],quantile(output,self.intv,axis=0))
            hig         = interp(depths,self.breaks[1:],quantile(output,1-self.intv,axis=0))
            SAR        = array([depths,low ,hig,mean1,median1])
            savetxt(self.hfol + self.dirt + '/' + self.Core + '/' + "SAR_{}_{}_{}.csv".format(self.Core, self.m,self.d_by), SAR.T, delimiter=',',fmt='%1.3f',header="depth,min,max,mean,median")
            #Plotting
            #SAR_depth   = figure(figsize=(10, 10))
            for sar in output:
                SAR_depth = plot(self.breaks[1:],sar, color='black',alpha = .02)
            xlabel('Depth')
            ylabel('yr BP')
            ylim(min(hig)-5,max(hig)+mean(hig)/2)
            #set_xlim([-self.by/12,self.breaks.max()+self.by/12])
            SAR_depth = plot(depths ,mean1, linestyle='dashed',c='red',alpha = .9)
            SAR_depth = plot(depths ,low, linestyle='dashed',c='red',alpha = .9)
            SAR_depth = plot(depths ,hig, linestyle='dashed',c='red',alpha = .9)
            savefig(self.hfol + self.dirt + '/' + self.Core + '/' + "SAR.pdf",bbox_inches = 'tight')
    
        else:
            print("rerun")
        
        
        
        
        
        
        
        
        
        
        


