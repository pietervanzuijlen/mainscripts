
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

def main(
         degree:  'polynomial degree for velocity'  = 3,
         mu:      'viscosity'                       = 1.,
         uref:    'number of uniform refinements'   = 3,
         M1:      'position of central circle'      = .5,
         npoints: 'number of sample points'         = 10,
         ):


    domain, geom = domainmaker.porous()
    ns = function.Namespace()

    domain = domain.refine(uref)

    ns.mu = mu 
    ns.x = geom
    ns.pbar = 1

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
    res += domain.boundary['left'].integral('g_i v_i d:x' @ ns, degree=degree*2)
    
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

    #res += dualspace.integral('(v_i,i - q) d:x' @ ns, degree=degree*2).derivative('dualtest')
    res += dualspace.boundary.integral('v d:x' @ ns, degree=degree*2).derivative('dualtest')

    #res1 = dualspace.integral('v_i^2 d:x' @ ns, degree=degree*2).derivative('dualtest')
    #res += dualspace.integral('(v_i v_i)^(1 / 2) d:x' @ ns, degree=degree*2).derivative('dualtest')

    gen_lhs_resnorm = solver.newton('dualtrail', res, constrain=cons)
    dualtrail = solver.solve(gen_lhs_resnorm)

    ns = ns(dualtrail=dualtrail)

    ns.Iz   = dualspace.project(ns.z, ns.ubasis, geometry=geom, ischeme='gauss4') 
    ns.Iz_i = 'ubasis_ni Iz_n'
    ns.Is   = dualspace.project(ns.s, ns.pbasis, geometry=geom, ischeme='gauss4') 
    ns.Is   = 'pbasis_n Is_n'
    plotter.plot_streamlines('velocity', domain, geom, ns, ns.u)
    plotter.plot_solution('pressure', domain, geom, ns.p)
    plotter.plot_streamlines('dualvector', dualspace, geom, ns, ns.z)
    plotter.plot_streamlines('Projected_dualvector', dualspace, geom, ns, ns.z-ns.Iz)
    plotter.plot_solution('dualscalar', dualspace, geom, ns.s)
    plotter.plot_solution('Projected_dualscalar', dualspace, geom, ns.s-ns.Is)

    ns.test = 'u_i,i - p'
    plotter.plot_solution('QuantityOfInterest', domain, geom, ns.test, npoints=10)

#    ns.test = 'u_i u_i'
#    plotter.plot_solution('test', domain, geom, ns.test, npoints=10)
#    ns.test = 'u_i <1 , 1>_i'
#    plotter.plot_solution('test', domain, geom, ns.test, npoints=10)
     
    anouncer.drum()

cli.run(main)
