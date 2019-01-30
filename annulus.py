'''
In this code different forms of adaptivity are tested on an lshaped domain on which a laplace problem is solved. The exact solution to this problem is known. Goal-oriented, residual-based and uniform refinement will be compared.
'''
from nutils import *
from utilities import *
import numpy as np
import sys

def main(degree      = 2,
         refinements = 1,
         npoints     = 5,
         num         = 0.5,
         uref        = 3,
         maxreflevel = 8,
         maxuref     = 5,
         beta        = 50,):

    sys.setrecursionlimit(9999)    

    methods = ['goaloriented','residualbased','uniform']
    methods = ['uniform']

    nelems    = {} 
    ndofs     = {} 
    error_velocity = {} 
    error_pressure = {} 

    for method in methods:

      nelems[method] = []
      ndofs[method]  = []
      error_velocity[method] = []
      error_pressure[method] = []

      domain, geom = domainmaker.annulus(uref=uref)

      for nref in range(refinements):

        print(method+' | '+str(nref))

        ns = function.Namespace()
        ns.x  = geom
        ns.mu = 1 
       
        # Elementwise length fraction
        areas = domain.integrate_elementwise(function.J(geom), degree=degree)
        ns.he = function.elemwise(domain.transforms, np.sqrt(areas))
        ns.beta = beta 
    
        # Exact solution
        ns.u0   = '.000001 x_0^2 x_1^4 (x_0^2 + x_1^2 - 1) (x_0^2 + x_1^2 - 16) (5 x_0^4 + 18 x_0^2 x_1^2 - 85 x_0^2 + 13 x_1^4 - 153 x_1^2 + 80)'
        ns.u1   = '-.000002 x_0 x_1^5 (x_0^2 + x_1^2 - 1) (x_0^2 + x_1^2 - 16) (5 x_0^4 - 51 x_0^2 + 6 x_0^2 x_1^2 - 17 x_1^2 + 16 + x_1^4)'

        ns.pexact   = '.0000001 x_0 x_1 (-x_1^2 + x_0^2) (x_0^2 + x_1^2 - 16)^2 (x_0^2 + x_1^2 - 1)^2 exp( 14 (x_0^2 + x_1^2)^(-0.5))'
        ns.uexact_i = '<u0 , u1>_i'
    
        # Body force
        ns.f_j = '- mu (uexact_i,j + uexact_j,i)_,i + pexact_,j'
            
        ######################
        ### Primal problem ###
        ######################

        ns.ubasis, ns.pbasis, ns.lbasis= function.chain([domain.basis('th-spline', degree=degree, continuity=degree-2).vector(2),
                                                         domain.basis('th-spline', degree=degree-1, continuity=degree-2),
                                                         [1,]])

        evalbasis = domain.basis('th-spline', degree=degree)
    
        # Trail functions
        ns.uh_i = 'ubasis_ni ?trail_n'
        ns.ph = 'pbasis_n ?trail_n'
        ns.lh = 'lbasis_n ?trail_n'
        
        # Test functions
        ns.v_i = 'ubasis_ni ?test_n'
        ns.q = 'pbasis_n ?test_n'
        ns.l = 'lbasis_n ?test_n'

        # Stress
        ns.stress_ij = 'mu (uh_i,j + uh_j,i) - ph δ_ij '
       
        # boundary condition
        cons = domain.boundary.project(0, onto=ns.ubasis, geometry=geom, degree=degree*2)

        # nitsche term
        ns.nitsche  = 'mu ( (uh_i,j + uh_j,i) n_i) v_j + mu ( (v_i,j + v_j,i) n_i ) uh_j - mu (beta / he) v_i uh_i - ph v_i n_i - q uh_i n_i'
    
        res = domain.integral('(f_i v_i - stress_ij v_i,j + q uh_l,l) d:x' @ ns, degree=degree*2)
        res += domain.integral('(- l ph - lh q ) d:x' @ ns, degree=degree*2)
        res += domain.boundary.integral('nitsche d:x' @ns, degree=degree*2)
        
        trail = solver.solve_linear('trail', res.derivative('test'))
        ns = ns(trail=trail)
    
        
        ####################
        ### Dual problem ###
        ####################

        dualdegree = degree + 1
    
        ns.zbasis, ns.sbasis, ns.Lbasis = function.chain([domain.basis('th-spline', degree=dualdegree, continuity=dualdegree-1).vector(2),
                                                          domain.basis('th-spline', degree=dualdegree-1, continuity=dualdegree-2),
                                                          [1,]])

        # Trail functions
        ns.z_i = 'zbasis_ni ?dualtrail_n'
        ns.s   = 'sbasis_n ?dualtrail_n'
        ns.lh  = 'Lbasis_n ?dualtrail_n'

        # Test functions
        ns.v_i = 'zbasis_ni ?dualtest_n'
        ns.q   = 'sbasis_n ?dualtest_n'
        ns.l   = 'Lbasis_n ?dualtest_n'

        # Shear traction
        ns.t_j = 'mu (v_i,j + v_j,i) n_i -  mu (v_i,l + v_l,i) n_i n_l n_j'

        # nitsche term
        ns.nitsche  = 'mu ( (z_i,j + z_j,i) n_i) v_j + mu ( (v_i,j + v_j,i) n_i ) z_j - mu (beta / he) v_i z_i - s v_i n_i - q z_i n_i'
    
        res = domain.boundary.integral('(t_i n_i) d:x' @ ns, degree=dualdegree*2)
        res += domain.integral('(- mu (z_j,i + z_i,j) v_j,i + s v_k,k + q z_l,l ) d:x' @ ns, degree=dualdegree*2)
        res += domain.integral('(- l s - lh q ) d:x' @ ns, degree=dualdegree*2)
        res += domain.boundary.integral('nitsche d:x' @ns, degree=degree*2)
    
        dualtrail = solver.solve_linear('dualtrail', res.derivative('dualtest'))
        ns = ns(dualtrail=dualtrail)

        ns.Iz   = domain.projection(ns.z, ns.ubasis, geometry=geom, degree=degree*2, constrain=cons)
        ns.Is   = domain.projection(ns.s, ns.pbasis, geometry=geom, degree=degree*2)

        # Get errors
        nelems[method]    += [len(domain)]
        ndofs[method]     += [len(evalbasis)]
        error_velocity[method] += [abs(domain.integrate(function.norm2('(uexact_i - uh_i) d:x' @ns), degree=degree*2))]
        error_pressure[method] += [abs(domain.integrate('(pexact - ph) d:x' @ns, degree=degree*2))]

        # indicators
        ns.moment = '(f_i + (uh_j,i + uh_i,j)_,j - ph_,i) (f_i + (uh_l,i + uh_i,l)_,l - ph_,i)'
        ns.zsharp = '(z_i - Iz_i) (z_i - Iz_i)'

        ns.incomp = '(uh_i,i)^2'
        ns.ssharp = '(s - Is)^2'

        h = np.sqrt(indicater.integrate(domain, geom, degree, 1, domain))
        moment = np.sqrt(indicater.integrate(domain, geom, degree, ns.moment, domain))
        incomp = np.sqrt(indicater.integrate(domain, geom, degree, ns.incomp, domain))
    
        z_int  = np.sqrt(indicater.integrate(domain, geom, degree, ns.zsharp, domain))
        s_int  = np.sqrt(indicater.integrate(domain, geom, degree, ns.ssharp, domain))

        plotter.plot_solution('pressure', domain, geom, ns.ph)
        plotter.plot_solution('exactpressure', domain, geom, ns.pexact)

        if method == 'goaloriented':
            
            indicators = incomp*s_int + moment*z_int 
            #plotter.plot_indicators(method+'contributions_'+str(nref), domain, geom, {'Momentum':moment,'z_sharp':z_int,'incompressibility':incomp,'s_sharp':s_int}, normalize=False)
            #plotter.plot_indicators(method+'indicators_'+str(nref), domain, geom, {'indicator':indicators,'incomp':incomp*s_int,'momentum':moment*z_int}, normalize=False)
            domain, refined = refiner.refine(domain, indicators, num, evalbasis, maxlevel=maxreflevel+uref, marker_type=None, select_type=None, refined_check=True)
            plotter.plot_mesh(method+'mesh'+str(nref), domain, geom)

        if method == 'residualbased':

            indicators = incomp*h + moment*h 
            #plotter.plot_indicators(method+'indicators_'+str(nref), domain, geom, {'indicator':indicators,'incomp':incomp*h,'momentum':moment*h}, normalize=False)
            domain, refined = refiner.refine(domain, indicators, num, evalbasis, maxlevel=maxreflevel+uref, marker_type=None, select_type=None, refined_check=True)
            plotter.plot_mesh(method+'mesh'+str(nref), domain, geom)

        if method == 'uniform':
            domain = domain.refine(1)
            refined = True
            if nref == maxuref:
                refined = False                    
            if not refined:
                break
    
        #### Post-processing ###

    #plotter.plot_convergence('Exact_error',ndofs,error_velocity,labels=['dofs','Exact velocity error'],slopemarker=True)
    #plotter.plot_convergence('Exact_error',ndofs,error_pressure,labels=['dofs','Exact pressure error'],slopemarker=True)
    #plotter.plot_convergence('Dofs_vs_elems',nelems,ndofs,labels=['nelems','ndofs'])
    
    plotter.plot_solution('pressure', domain, geom, ns.ph)
    plotter.plot_streamlines('velocity', domain, geom, ns, ns.uh)
    #plotter.plot_solution('dual_pressure', domain, geom, ns.s)
    #plotter.plot_streamlines('dual_velocity', domain, geom, ns, ns.z)
    #plotter.plot_streamlines('traction', domain, geom, ns, ns.traction)

    #ns.gradp_i = 'pexact_,i'
    #ns.gradph_i = 'ph_,i'
    #plotter.plot_streamlines('gradpressureexact', domain, geom, ns, ns.gradp)
    #plotter.plot_streamlines('gradpressure', domain, geom, ns, ns.gradph)
    #plotter.plot_streamlines('body_force', domain, geom, ns, ns.f)
    #plotter.plot_streamlines('als_body_force', domain, geom, ns, ns.falt)

with config(verbose=3,nprocs=6):
    cli.run(main)