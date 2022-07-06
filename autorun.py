#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 09:53:31 2022

Autorun PyPlum for testing

@author: maquinolopez
"""
# import PyPlum 

# core = PyPlum.Plum(iterations=500,burnin=1000,thi=1)

# core.runPlum()

# coreg = PyPlum.Plum("coreg",mean_acc=50, \
#     reservoir_eff=True,r_effect_prior=187,r_effect_psd=18,\
#     Al=.01,thick=.5)


# coreg.runPlum()


import PyPlum 

test = PyPlum.Plum()

#test.runPlum()

test.SAR_d()

