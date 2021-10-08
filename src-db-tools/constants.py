#!/bin/env python
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------
# Static information

sex_name2covariate = dict(female = -0.5, both = 0.0, male = +0.5)
sex_name2dismod_id = dict(male = 1, female = 2, both = 3)
sex_dismod_id2_name = {v:k for k,v in sex_name2dismod_id.items()}
sex_name2ihme_id = dict(both = 0, male = 1, female = 2)
sex_ihme_id2_name = {v:k for k,v in sex_name2ihme_id.items()}

rates = ('pini', 'iota', 'rho', 'chi', 'omega')

integrand2rate = {'prevalence' : 'pini',
                  'Sincidence' : 'iota',
                  'incidence' : 'iota',
                  'remission' : 'rho',
                  'mtexcess' : 'chi',
                  'mtother' : 'omega'}

rate2integrand = { 'pini' : 'prevalence', 
                   'iota' : 'incidence',
                   'rho'  : 'remission',
                   'chi'  : 'mtexcess',
                   'omega': 'mtall' }

rate2Sintegrand = { 'pini' : 'prevalence', 
                    'iota' : 'Sincidence',
                    'rho'  : 'remission',
                    'chi'  : 'mtexcess',
                    'omega': 'mtall' }
_time_window_for_fit_ = 2.5001

mortality_integrands = ('mtall', 'mtexcess', 'mtother', 'mtspecific', 'mtstandard', 'mtwith')
integrands = tuple(['Sincidence', 'Tincidence', 'prevalence', 'relrisk', 'remission', 'susceptible', 'withC'] + list(mortality_integrands))
