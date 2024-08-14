# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:38:26 2022

@author: ions
"""




import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from statsmodels.stats.proportion import proportion_confint
import math


class Fit_thermal_state:
    
    def __init__(self,t,pop,bounds = [] ,p0 = [] , num_shots = 100, sideband = 'BSB'):
        self.t = t
        self.pop = pop
        self.bounds = bounds
        self.p0 = p0
        self.num_shots = num_shots
        self.fit = None
        self.fit_cov = None
        self.sideband = sideband
        self.get_std(self.pop, self.num_shots)
        
        
    def thermal_state_bsb(self,t,nbar,Ω,η,γ):
        Pb = 0
            
        for n in range(800):
            Pn = (1/(nbar+1))*(nbar/(1+nbar))**n
            Ωnnp1 = np.sqrt(n+1)*η*Ω
            γn = γ*np.sqrt(n+1)
            
            Pb+= Pn*np.cos(Ωnnp1*t)*np.exp(-γn*t)
            
            
        return 1-0.5*(1+Pb)

    
    def thermal_state_rsb(self,t,nbar,Ω,η,γ):
        Pb = 0
            
        for n in range(800):
            Pn = (1/(nbar+1))*(nbar/(1+nbar))**n
            Ωnnp1 = np.sqrt(n+1)*η*Ω
            γn = γ*np.sqrt(n+1)
            
            Pb+= Pn*np.cos(Ωnnp1*t)*np.exp(-γn*t)
            
            
        return 1-0.5*(1+Pb)
    
    def fit_thermal_state(self):
        bounds = self.bounds
        p0 = self.p0
        t = self.t
        pop = self.pop
        
        if self.sideband == 'BSB':
            if len(bounds) >0 and len(p0)>0:
                fit_bsb, fit_bsb_cov = curve_fit(self.thermal_state_bsb,t,pop, bounds = bounds, p0 = p0,sigma = self.std)
            elif len(bounds)>0 and len(p0)==0:
                fit_bsb, fit_bsb_cov = curve_fit(self.thermal_state_bsb,t,pop, bounds = bounds,sigma = self.std)
            else:
                fit_bsb, fit_bsb_cov = curve_fit(self.thermal_state_bsb,t,pop,sigma = self.std)
        else:
            if len(bounds) >0 and len(p0)>0:
                fit_bsb, fit_bsb_cov = curve_fit(self.thermal_state_rsb,t,pop, bounds = bounds, p0 = p0,sigma = self.std)
            elif len(bounds)>0 and len(p0)==0:
                fit_bsb, fit_bsb_cov = curve_fit(self.thermal_state_rsb,t,pop, bounds = bounds,sigma = self.std)
            else:
                fit_bsb, fit_bsb_cov = curve_fit(self.thermal_state_rsb,t,pop,sigma = self.std)
        
        
        self.fit = fit_bsb
        self.fit_cov = fit_bsb_cov
        return fit_bsb, fit_bsb_cov
    
    def plot_fit(self,color):
        if self.fit:
            fit = self.fit
            fit_cov = self.fit_cov
        else:
            self.fit_thermal_state()
            fit = self.fit
            fit_cov = self.fit_cov
        print(self.fit)
        t = self.t
        pop = self.pop
        std = self.std
        t_fit = np.linspace(min(t),max(t),1000)
        plt.errorbar(t*1e6,pop,std,marker = '.', ls = 'none',c= color)
        fit_std = np.sqrt(np.diag(fit_cov))[0]
        if self.sideband =='BSB':
            plt.plot(t_fit*1e6,self.thermal_state_bsb(t_fit,*self.fit),label = r'nbar:{:.1f}$({:.0f})$'.format(self.fit[0],10*fit_std),c = color)
        else:
            plt.plot(t_fit*1e6,self.thermal_state_rsb(t_fit,*self.fit),label = r'nbar:{:.1f}$({:.0f})$'.format(self.fit[0],10*fit_std),c = color)
        plt.xlabel('BSB pulse duration (μs)')
        plt.ylabel(r'$P_{\uparrow}$',fontsize = 14)
        plt.legend();
        
    def get_std(self,prob,N):
        std = []
        for p in prob:
            k = p*N
            confint = proportion_confint(k, N, alpha=0.3173, method='beta')
            uncertainty = (confint[1]-confint[0])/2
            std.append(uncertainty)
        self.std = std
        