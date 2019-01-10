'''
steps to be taken:

- Make dual space by order elevation possible (test efficiency)
- Add skeleton and ghost to dual problem
- Define goal residual and indicators
- Add adaptivity

'''

from   nutils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections

from utilities import *

def main(
         refinements= 5,
         degree     = 2,
         npoints    = 5,
         nelem      = 4,
         maxrefine  = 3,
         uref       = 2,
         beta       = 0.5,
         num        = 0.5,
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


    for nref in range(refinements):

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
    
        ns = function.Namespace()
    
        ns.mu   = 1 
        ns.pbar = 1
        ns.x    = geom
        ns.beta = beta
    
        ### Primal problem
       
        # Define bases
        ns.ubasis, ns.pbasis = function.chain([domain.basis('th-spline', degree=degree, continuity=degree-1).vector(2),
                                               domain.basis('th-spline', degree=degree, continuity=degree-1)])
    
        # Get h_e 
        areas = domain.integrate_elementwise(function.J(geom), degree=degree)
        ns.h  = np.mean(np.sqrt(areas))
        
        # Trail functions
        ns.u_i = 'ubasis_ni ?trail_n'
        ns.p = 'pbasis_n ?trail_n'
        
        # Test functions
        ns.v_i = 'ubasis_ni ?test_n'
        ns.q = 'pbasis_n ?test_n'
        
        ns.stress_ij = 'mu (u_i,j + u_j,i) - p δ_ij'
        ns.g_n = 'n_n'
    
        # nitsche term
        ns.nitsche  = '-mu ( (u_i,j + u_j,i) n_i) v_j + ( (v_i,j + v_j,i) n_i ) u_j + mu (beta / h) v_i u_i + p v_i n_i - q u_i n_i'
    
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
    
        ns.gskel  = 0.05
        ns.gghost = 0.005
        
        ns.skeleton = 'gskel mu^-1 h^{} [[{}]] [[{}]]'.format(2*degree+1, jumpp, jumpq)
        ns.ghost    = 'gghost  mu h^{} [[{}]] [[{}]]'.format(2*degree-1, jumpu, jumpv)
    
        res  = domain.integral('(stress_ij v_i,j + q u_l,l) d:x' @ ns, degree=degree*2)
        res += domain.boundary['top,bottom,trimmed'].integral('nitsche d:x' @ns, degree=degree*2)
        res += domain.boundary['left'].integral('(g_i v_i) d:x' @ ns, degree=degree*2)
        res += skeleton.integral('skeleton d:x' @ns, degree=degree*2)
        res += ghost.integral('ghost d:x' @ns, degree=degree*2)
                
    
        trail = solver.solve_linear('trail', res.derivative('test'))
        ns = ns(trail=trail) 
    
    
        ### Dual problem
        dualdegree = degree + 1
    
        # Define bases
        ns.zbasis, ns.sbasis = function.chain([domain.basis('th-spline', degree=dualdegree, continuity=dualdegree-1).vector(2),
                                               domain.basis('th-spline', degree=dualdegree, continuity=dualdegree-1)])
    
        # Trail functions
        ns.z_i = 'zbasis_ni ?dualtrail_n'
        ns.s = 'sbasis_n ?dualtrail_n'
        
        # Test functions
        ns.v_i = 'zbasis_ni ?dualtest_n'
        ns.q = 'sbasis_n ?dualtest_n'
        
        # nitsche term
        ns.nitsche  = '-mu ( (z_i,j + z_j,i) n_i) v_j + ( (v_i,j + v_j,i) n_i ) z_j + mu (beta / h) v_i z_i - s v_i n_i + q z_i n_i'
    
        #ns.nitsche  = '-mu ( (u_i,j + u_j,i) n_i) v_j + ( (v_i,j + v_j,i) n_i ) u_j + mu (beta / h) v_i u_i + p v_i n_i + q u_i n_i'
        # Getting skeleton and ghost stabilization terms
        norm_derivative  = '({val}_,{i} n_{i})'
        jumps = '(s_,a n_a)'
        jumpz = '(z_n,a n_a)'
        for i in 'bcdef'[:(dualdegree-1)]:
            jumps = norm_derivative.format(val=jumps, i=i)
            jumpz = norm_derivative.format(val=jumpz, i=i)
    
        jumpq = '(q_,g n_g)'
        jumpv = '(v_n,g n_g)'
        for i in 'hijkl'[:(dualdegree-1)]:
            jumpq = norm_derivative.format(val=jumpq, i=i)
            jumpv = norm_derivative.format(val=jumpv, i=i)
    
        ns.gskel  = 0.05
        ns.gghost = 0.005
        
        ns.skeleton = 'gskel mu^-1 h^{} [[{}]] [[{}]]'.format(2*dualdegree+1, jumpq, jumps)
        ns.ghost    = 'gghost  mu h^{} [[{}]] [[{}]]'.format(2*dualdegree-1, jumpv, jumpz)
    
        # Quantity of interest: outflow
        res = domain.integral('(( mu (v_i,j + v_j,i) - q δ_ij ) z_i,j + s v_l,l ) d:x' @ ns, degree=dualdegree*2)
        res += domain.boundary['top,bottom,trimmed'].integral('nitsche d:x' @ns, degree=dualdegree*2)
        res += domain.boundary['right'].integral('(n_i v_i) d:x' @ ns, degree=dualdegree*2)
        res += skeleton.integral('skeleton d:x' @ns, degree=dualdegree*2)
        res += ghost.integral('ghost d:x' @ns, degree=dualdegree*2)
                
        dualtrail = solver.solve_linear('dualtrail', res.derivative('dualtest'))
        ns = ns(dualtrail=dualtrail) 
    
        ns.Iz   = domain.projection(ns.z, ns.ubasis, geometry=geom, degree=dualdegree*2)
        ns.Is   = domain.projection(ns.s, ns.pbasis, geometry=geom, degree=dualdegree*2)
    
        indicators, force_ind, incom_ind, bound_ind, jump_ind, z_ind, s_ind = get_indicators(domain, geom, ns, dualdegree)

        contributions = {'Force': force_ind.indicators, 'Incompressibility': incom_ind.indicators, 'In- and outflow': bound_ind.indicators, 'Interface': jump_ind.indicators, '||z-Iz||': z_ind.indicators, '||s-Is||': s_ind.indicators}
    
    
        # plotting
        plotter.plot_indicators('Indicators_'+str(nref),domain, geom, {'Elem sizes':indicators.indicators})
        plotter.plot_indicators('Contributions_'+str(nref),domain, geom, contributions)
        plotter.plot_solution('sharp_dual_velocity_'+str(nref), domain, geom, function.norm2(ns.z-ns.Iz))
        plotter.plot_solution('sharp_dual_pressure_'+str(nref), domain, geom, ns.s-ns.Is)

        domain = refiner.fractional_marking(domain, indicators, num)
        plotter.plot_mesh('mesh_'+str(nref), domain, geom)

    anouncer.drum()

def get_indicators(domain, geom, ns, dualdegree):

    ns.inflow_i  = 'g_i + stress_ij n_j'
    ns.jump_i    = '[-stress_ij] n_j '
    ns.force_i   = 'stress_ij,j'
    ns.incom     = 'u_i,i'
    ns.outflow_i = '- stress_ni n_n'
    ns.sharpz_i    = 'z_i - Iz_i'
    ns.sharps    = 's - Is'

    inflow  = function.norm2(ns.inflow)
    jump    = function.norm2(ns.jump)
    force   = function.norm2(ns.force)
    outflow = function.norm2(ns.outflow)
    sharpz  = function.norm2(ns.sharpz)
    incom   = function.abs(ns.incom)
    sharps  = function.abs(ns.sharps)

    indicators = indicater.elementbased(domain, geom, dualdegree, dualspacetype='k-refined')
    
    force_ind = indicater.elementbased(domain, geom, dualdegree, dualspacetype='k-refined')
    incom_ind = indicater.elementbased(domain, geom, dualdegree, dualspacetype='k-refined')
    bound_ind = indicater.elementbased(domain, geom, dualdegree, dualspacetype='k-refined')
    jump_ind = indicater.elementbased(domain, geom, dualdegree, dualspacetype='k-refined')
    z_ind = indicater.elementbased(domain, geom, dualdegree, dualspacetype='k-refined')
    s_ind = indicater.elementbased(domain, geom, dualdegree, dualspacetype='k-refined')

    indicators.goaloriented(domain, incom*sharps, 'internal')
    indicators.goaloriented(domain, force*sharpz, 'internal')
    indicators.goaloriented(domain.interfaces, jump*sharpz, 'interface')
    indicators.goaloriented(domain.boundary['left'], inflow*sharpz, 'boundary')
    indicators.goaloriented(domain.boundary['right'], outflow*sharpz, 'boundary')

    force_ind.goaloriented(domain, incom*sharps, 'internal')
    incom_ind.goaloriented(domain, force*sharpz, 'internal')
    jump_ind.goaloriented(domain.interfaces, jump*sharpz, 'interface')
    bound_ind.goaloriented(domain.boundary['left'], inflow*sharpz, 'boundary')
    bound_ind.goaloriented(domain.boundary['right'], outflow*sharpz, 'boundary')

    z_ind.goaloriented(domain, sharpz, 'internal')
    s_ind.goaloriented(domain, sharps, 'internal')

    return indicators, force_ind, incom_ind, bound_ind, jump_ind, z_ind, s_ind

with config(verbose=3,nprocs=6):
    cli.run(main)
