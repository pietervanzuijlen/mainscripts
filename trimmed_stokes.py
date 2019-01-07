
'''
In this code different forms of adaptivity are tested on a domain on which the stokes problem is solved. Goal-oriented, residual-based and uniform refinement will be compared.

'''

from nutils import *
from utilities import *
from matplotlib import collections

import numpy as np
import matplotlib.pyplot as plt


def main(
         degree     = 3,
         npoints    = 10,
         uref       = 2,
         maxrefine  = 2,
         nelem      = 4,
         beta       = 0.5,
         gskel      = 0.05,
         gghost     = 0.005,
         ):

    # Defining grid 
    grid, geom = mesh.rectilinear([numpy.linspace(0,1,nelem+1),numpy.linspace(0,1,nelem+1)])
    grid = grid.refine(uref)

    # Defining trimmed domain
    rm = .2
    rc = .2
    M0 = .5
    M1 = .5

    x0, x1 = geom
    
    domain = grid.trim(function.norm2((x0,x1))-rc, maxrefine = maxrefine)
    domain = domain.trim(function.norm2((x0-1,x1))-rc, maxrefine = maxrefine)
    domain = domain.trim(function.norm2((x0-1,x1-1))-rc, maxrefine = maxrefine)
    domain = domain.trim(function.norm2((x0,x1-1))-rc, maxrefine = maxrefine)
    domain = domain.trim(function.norm2((x0-M0,x1-M1))-rm, maxrefine = maxrefine)

    domain = domain.refined_by((domain.elements[1].transform,))

    # Defining the background and skeleton 
    background_elems = []

    for elem in grid:
        trans = transform.lookup(elem.transform,domain.edict)
        if trans:
            background_elems += [elem]

    background = grid.subset(background_elems)
    skeleton = background.interfaces
    ghost = []

    # Defining ghost
    for iface in skeleton:
        for trans in iface.transform, iface.opposite:
        
            # find the corresponding element in the background mesh
            ielemb, tailb = transform.lookup_item(trans,background.edict)
            belem = background.elements[ielemb]
  
            # find the corresponding element in the trimmed mesh
            ielemt, tailt = transform.lookup_item(trans,domain.edict)
            telem = domain.elements[ielemt]
  
            if belem != telem:
              assert belem.transform == telem.transform
              assert belem.opposite  == telem.opposite
              ghost.append(iface)
              break

    ghost = topology.UnstructuredTopology(background.ndims-1, ghost)


    # Creating namespace
    ns = function.Namespace()

    ns.mu     = 1 
    ns.pbar   = 1
    ns.x      = geom
    ns.beta   = beta
    ns.gskel  = gskel
    ns.gghost = gghost
   
    # Define bases on same space
    ns.ubasis, ns.pbasis = function.chain([domain.basis('th-spline', degree=degree, continuity=degree-1).vector(2),
                                           domain.basis('th-spline', degree=degree, continuity=degree-1)])

    # Get h_e 
    areas = domain.integrate_elementwise(function.J(geom), degree=degree)
    gridareas = grid.integrate_elementwise(function.J(geom), degree=degree)

    h_K = np.sqrt(np.mean(gridareas))

    he_map = {}
    hF_map = {}
    for elem, area in zip(domain, areas):
        trans = elem.transform
        he_map[trans] = np.sqrt(area)
        head, tail = transform.lookup(trans, domain.edict)
        hF_map[elem.transform] = h_K*0.5**(len(head)-2-uref)

    ns.he = function.elemwise(he_map, ())
    ns.hF = function.elemwise(hF_map, ())

    # Inflow traction
    ns.g_n = 'n_n'

    # Trail functions
    ns.u_i = 'ubasis_ni ?trail_n'
    ns.p = 'pbasis_n ?trail_n'
    
    # Test functions
    ns.v_i = 'ubasis_ni ?test_n'
    ns.q = 'pbasis_n ?test_n'

    # Operators
    A = 'mu ({u}_i,j + {u}_j,i) {v}_i,j'
    B = '-{p} {v}_i,i' 
    F = 'g_i n_i'

    N_A = '-mu ( ({u}_i,j + {u}_j,i) n_i) {v}_j -mu ( ({v}_i,j + {v}_j,i) n_i ) {u}_j + mu (beta / he) {v}_i {u}_i'
    N_B = '{p} {v}_i n_i'

    S = 'gskel mu^-1 hF^{} [[{}]] [[{}]]'
    G = 'gghost mu hF^{} [[{}]] [[{}]]'
    
    #ns.stress_ij = 'mu (u_i,j + u_j,i) - p δ_ij'

    # nitsche term
    #ns.nitsche  = '-mu ( (u_i,j + u_j,i) n_i) v_j + ( (v_i,j + v_j,i) n_i ) u_j + mu (beta / he) v_i u_i + p v_i n_i + q u_i n_i'

    # Getting skeleton and ghost stabilization terms
    norm_derivative  = '({val}_,{i} n_{i})'

    jumpp = '(p_,a n_a)'
    jumpu = '(u_n,a n_a)'
    for i in 'bcdef'[:(degree-1)]:
        jumpp = norm_derivative.format(val=jumpp, i=i)
        jumpu = norm_derivative.format(val=jumpu, i=i)
        
    jumpq = '(q_,g n_g)'
    jumpv = '(v_n,g n_g)'
    for i in 'hijkl'[:(degree-1)]:
        jumpq = norm_derivative.format(val=jumpq, i=i)
        jumpv = norm_derivative.format(val=jumpv, i=i)
    
    ns.skeleton = 'gskel mu^-1 hF^{} [[{}]] [[{}]]'.format(2*degree+1, jumpp, jumpq)
    ns.ghost    = 'gghost  mu hF^{} [[{}]] [[{}]]'.format(2*degree-1, jumpu, jumpv)

    res  = domain.integral('(stress_ij v_i,j + q u_l,l) d:x' @ ns, degree=degree*2)
    res += domain.boundary['top,bottom,trimmed'].integral('nitsche d:x' @ns, degree=degree*2)
    res += domain.boundary['left'].integral('(g_i v_i) d:x' @ ns, degree=degree*2)
    res += skeleton.integral('skeleton d:x' @ns, degree=degree*2)
    res += ghost.integral('ghost d:x' @ns, degree=degree*2)
        
    trail = solver.solve_linear('trail', res.derivative('test'))
    ns = ns(trail=trail) 

    plotter.plot_streamlines('velocity', domain, geom, ns, ns.u)
    plotter.plot_solution('pressure', domain, geom, ns.p)
    plotter.plot_solution('hefunc', domain, geom, ns.he, alpha=1)
    plotter.plot_solution('hFfunc', domain, geom, ns.hF, alpha=1)

with config(verbose=3,nprocs=6):
    cli.run(main)
