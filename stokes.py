'''
In this code different forms of adaptivity are tested on a domain on which the stokes problem is solved. Goal-oriented, residual-based and uniform refinement will be compared.

'''

from   nutils import *
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import collections
from utilities import writer
from utilities import plotter
from utilities import domainmaker
from utilities import indicater
from utilities import anouncer
from utilities import refiner

import treelog

def main(
         degree:  'polynomial degree for velocity'  = 3,
         mu:      'viscosity'                       = 1.,
         maxref:  'number of refinement iterations' = 30,
         maxuref: 'maximum uniform refinements'     = 3,
         M1:      'position of central circle'      = .35,
         write:   'write results to file'           = True,
         npoints: 'number of sample points'         = 5,
         num:     'to be refined fraction'          = 85,
         uref:    'number of uniform refinements'   = 2,
         ):

  mid = M1
  datalog = treelog.DataLog('../results/stokes/images')

  methods = ['residual','goal','uniform']

  with treelog.add(datalog):

        for method in methods:
    
            domain, geom = domainmaker.porous(uref=uref, M1=mid)
            ns = function.Namespace()
    
            ns.mu = mu 
            ns.x = geom
            ns.pbar = 1
            
            maxlvl       = []
            residual     = []
            sum_residual = []
            sum_goal     = []
            nelems       = []
    
            for nref in range(maxref):
    
                # Primal problem
    
                ns.ubasis, ns.pbasis = function.chain([domain.basis('th-spline', degree=degree, patchcontinuous=True, continuity=degree-2).vector(2),
                                                       domain.basis('th-spline', degree=degree-1, patchcontinuous=True, continuity=degree-2)])
    
                # Trail functions
                ns.u_i = 'ubasis_ni ?trail_n'
                ns.p = 'pbasis_n ?trail_n'
                
                # Test functions
                ns.v_i = 'ubasis_ni ?test_n'
                ns.q = 'pbasis_n ?test_n'
    
                ns.stress_ij = 'mu (u_i,j + u_j,i) - p δ_ij'
                ns.g_n = 'n_n'
                            
                # boundary condition
                sqr = domain.boundary['top,bottom,corners,circle'].integral('u_k u_k d:x' @ ns, degree=degree*2)
                cons = solver.optimize('trail', sqr, droptol=1e-15)
    
                res = domain.integral('(stress_ij v_i,j + q u_l,l) d:x' @ ns, degree=degree*2)
                res += domain.boundary['left'].integral('(g_i v_i) d:x' @ ns, degree=degree*2)
                
                trail = solver.solve_linear('trail', res.derivative('test'), constrain=cons)
                
                ns = ns(trail=trail) 
                
                # Dual problem
                dualspace = domain.refine(1)
    
                ns.dualubasis, ns.dualpbasis = function.chain([dualspace.basis('th-spline', degree=degree, patchcontinuous=True, continuity=degree-2).vector(2),
                                                               dualspace.basis('th-spline', degree=degree-1, patchcontinuous=True, continuity=degree-2)])
        
                # Trail functions
                ns.z_i = 'dualubasis_ni ?dualtrail_n'
                ns.s = 'dualpbasis_n ?dualtrail_n'
    
                # Test functions
                ns.v_i = 'dualubasis_ni ?dualtest_n'
                ns.q = 'dualpbasis_n ?dualtest_n'
    
                sqr = dualspace.boundary['top,bottom,corners,circle'].integral('z_k z_k d:x' @ ns, degree=degree*2)
                cons = solver.optimize('dualtrail', sqr, droptol=1e-15)
    
                res = dualspace.integral('(( mu (v_i,j + v_j,i) - q δ_ij ) z_i,j - s v_i,i ) d:x' @ ns, degree=degree*2).derivative('dualtest')

                # Quantity of interest: outflow
                res += dualspace.boundary['left'].integral('( -v_i n_i ) d:x' @ ns, degree=degree*2).derivative('dualtest')

                # Quantity of interest: drag coefficient
                #res += dualspace.boundary['circle'].integral(' n_i (mu (v_i,j + v_j,i) - q δ_ij) <1 , 0>_j d:x' @ ns, degree=degree*2).derivative('dualtest')

                # Quantity of interest: lift coefficient
                #res += dualspace.boundary['circle'].integral(' n_i (mu (v_i,j + v_j,i) - q δ_ij) <0 , 1>_j d:x' @ ns, degree=degree*2).derivative('dualtest')
    
                dualtrail = solver.solve_linear('dualtrail', res, constrain=cons)
    
                ns = ns(dualtrail=dualtrail)
    
                ns.Iz   = dualspace.project(ns.z, ns.ubasis, geometry=geom, ischeme='gauss4') 
                ns.Iz_i = 'ubasis_ni Iz_n'
                ns.Is   = dualspace.project(ns.s, ns.pbasis, geometry=geom, ischeme='gauss4') 
                ns.Is   = 'pbasis_n Is_n'
    
                # Getting the indicators
                residual_indicators, res_int, res_jump, res_bound = elem_errors_residual(ns, geom, domain, dualspace, degree) 
                goal_indicators, goal_inter, goal_inflow = elem_errors_goal(ns, geom, domain, dualspace, degree) 
    
                #residual_indicators = func_errors_residual(ns, geom, domain, degree) 
                #goal_indicators = func_errors_goal(ns, geom, domain, dualspace, degree) 
    
                # Building lists with error data
                try:
                    maxlvl   += [len(domain.levels)]
                except:
                    maxlvl   += [1]
                residual     += [domain.integrate(function.norm2('(-mu (u_i,jj + u_j,ij) + p_,i) d:x' @ns)+function.abs('u_i,i d:x' @ns), ischeme='gauss5')]
                sum_residual += [sum(residual_indicators.indicators.values())]
                sum_goal     += [sum(goal_indicators.indicators.values())]
                nelems       += [len(domain)]
    

                # Refining and plotting 
                if method == 'residual':
                    indicators = {'Indicators':residual_indicators.indicators,'Internal':res_int.indicators,'Interfaces':res_jump.indicators,'Boundary':res_bound.indicators}
                    plotter.plot_indicators(method+'_indicators'+str(nref),domain, geom, indicators)
                    domain = refiner.refine(domain, residual_indicators, num)
                    
                if method == 'goal':
                    indicators = {'Indicators':goal_indicators.indicators,'Internal':goal_inter.indicators,'Boundary':goal_inflow.indicators}
                    plotter.plot_indicators(method+'_indicators'+str(nref),domain, geom, indicators)
                    domain = refiner.refine(domain, goal_indicators, num)
                    
                if method == 'uniform':
                    domain = domain.refine(1)
                    if nref == maxuref:
                        break
    
                plotter.plot_mesh('mesh_'+method+str(mid)+'_'+str(nref),domain,geom)
        
            if write:
                writer.write('../results/stokes'+method+str(mid),
                            {'degree': degree, 'nref': maxref, 'refinement number': num},
                              maxlvl       = maxlvl,
                              residual     = residual,
                              sum_residual = sum_residual,
                              sum_goal     = sum_goal,
                              nelems       = nelems,)
    
        plotter.plot_streamlines('velocity'+method+str(mid), domain, geom, ns, ns.u)
        plotter.plot_solution('pressure'+method+str(mid), domain, geom, ns.p)
        plotter.plot_streamlines('dualvector'+method+str(mid), dualspace, geom, ns, ns.z)
        plotter.plot_solution('dualscalar'+method+str(mid), dualspace, geom, ns.s)

def func_errors_residual(ns, geom, domain, degree):

    indicators = indicater.functionbased(domain, geom, ns.evalbasis, degree)
    
    ns.momentum_j = '- mu (u_i,ji + u_j,ii) + p_,j'
    ns.incompress = 'u_i,i'    

    indicators = indicators.add(domain, ns.evalbasis, function.norm2(ns.momentum))
    indicators = indicators.add(domain, ns.evalbasis, function.abs(ns.incompress))
    indicators = indicators.abs()

    return indicators

def func_errors_goal(ns, geom, domain, dualspace, degree):

    indicators = indicater.functionbased(dualspace, geom, ns.evalbasis, degree)
    
    ns.inter = 'mu (u_i,j + u_j,i) (z_i,j - Iz_i,j) - p (z_i,i - Iz_i,i)'
    ns.bound = 'g_i (z_i - Iz_i)'
    ns.incom = '(s - Is) u_i,i'

    indicators = indicators.add(dualspace, ns.evalbasis, ns.inter)
    indicators = indicators.add(dualspace.boundary['left'], ns.evalbasis, ns.bound)
    indicators = indicators.abs()
    indicators = indicators.add(dualspace, ns.evalbasis, function.abs(ns.incom))
    indicators = indicators.abs()

    return indicators

def elem_errors_residual(ns, geom, domain, dualspace, degree):

    ns.inflow_i  = 'g_i + stress_ij n_j'
    ns.jump_i    = '[-stress_ij] n_j '
    ns.force_i   = 'stress_ij,j'
    ns.incom     = 'u_i,i'
    ns.outflow_i = '- stress_ni n_n'

    inflow  = function.norm2(ns.inflow)*function.J(geom)
    jump    = function.norm2(ns.jump)*function.J(geom)
    force   = function.norm2(ns.force)*function.J(geom)
    incom   = function.abs(ns.incom)*function.J(geom)
    outflow = function.norm2(ns.outflow)*function.J(geom)

    residual_indicators = indicater.elementbased(domain, geom, degree)
    int_indicators      = indicater.elementbased(domain, geom, degree)
    jump_indicators     = indicater.elementbased(domain, geom, degree)
    bound_indicators    = indicater.elementbased(domain, geom, degree)

    residual_indicators = residual_indicators.residualbased(domain, force, 'internal')
    residual_indicators = residual_indicators.residualbased(domain, incom, 'internal')
    int_indicators      = int_indicators.residualbased(domain, force, 'internal')
    int_indicators      = int_indicators.residualbased(domain, incom, 'internal')

    residual_indicators = residual_indicators.residualbased(domain.interfaces, jump, 'interface')
    jump_indicators     = jump_indicators.residualbased(domain.interfaces, jump, 'interface')

    residual_indicators = residual_indicators.residualbased(domain.boundary['left'], inflow, 'boundary')
    bound_indicators    = bound_indicators.residualbased(domain.boundary['left'], inflow, 'boundary')

    residual_indicators = residual_indicators.residualbased(domain.boundary['right'], outflow, 'boundary')
    bound_indicators    = bound_indicators.residualbased(domain.boundary['right'], outflow, 'boundary')

    return residual_indicators, int_indicators, jump_indicators, bound_indicators


 
def elem_errors_goal(ns, geom, domain, dualspace, degree):

    ns.inflow = 'g_i (z_i - Iz_i)'
    ns.inter  = '-mu (u_i,j + u_j,i) (z_i,j + Iz_i,j) + p (z_i,i + Iz_i,i)'

    inflow = ns.inflow*function.J(geom)
    inter = ns.inter*function.J(geom)
    
    goal_indicators   = indicater.elementbased(domain, geom, degree)
    inflow_indicators = indicater.elementbased(domain, geom, degree)
    inter_indicators  = indicater.elementbased(domain, geom, degree)

    goal_indicators   = goal_indicators.goaloriented(dualspace.boundary['left'], inflow, 'boundary')
    inflow_indicators   = inflow_indicators.goaloriented(dualspace.boundary['left'], inflow, 'boundary')

    goal_indicators  = goal_indicators.goaloriented(dualspace, inter, 'internal')
    inter_indicators  = inter_indicators.goaloriented(dualspace, inter, 'internal')

    goal_indicators = goal_indicators.abs()

    return goal_indicators, inter_indicators, inflow_indicators

 
with config(verbose=3,nprocs=6):
    cli.run(main)

