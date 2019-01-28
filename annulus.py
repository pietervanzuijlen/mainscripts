'''
In this code different forms of adaptivity are tested on an lshaped domain on which a laplace problem is solved. The exact solution to this problem is known. Goal-oriented, residual-based and uniform refinement will be compared.
'''
from nutils import *
from utilities import *
import numpy as np
import sys

def main(degree      = 3,
         maxref      = 5,
         npoints     = 5,
         num         = 0.5,
         uref        = 1,
         maxreflevel = 5,
         maxuref     = 5,):

    sys.setrecursionlimit(9999)    
    methods = ['goaloriented','residualbased','uniform']

    for method in methods:

      domain, geom = domainmaker.annulus(uref=uref)

      for nref in range(maxref):

        print(method+' | '+str(nref))

        # Exact solution
        ns = function.Namespace()
        arguments = {}
    
        ns.x   = geom
        ns.mu  = 1
        ns.beta= 5
        ns.u0  = '.000001 x_0^2 x_1^4 (x_0^2 + x_1^2 - 1) (x_0^2 + x_1^2 - 16) (5 x_0^4 + 18 x_0^2 x_1^2 - 85 x_0^2 + 13 x_1^4 - 153 x_1^2 + 80)'
        ns.u1  = '.000001 x_0 x_1^5 (x_0^2 + x_1^2 - 1) (x_0^2 + x_1^2 - 16) (102 x_0^2 + 34 x_1^2 - 10 x_0^4 - 12 x_0^2 x_1^2 - 2 x_1^4 - 32)'
        ns.pexact   = '.0000001 x_0 x_1 (x_1^2 - x_0^2) (x_0^2 + x_1^2 - 16)^2 (x_0^2 + x_1^2 - 1)^2 exp( 14 (x_0^2 + x_1^2)^(-0.5))'
        
        ns.uexact_i = '<u0 , u1>_i'
    
        # Body force
        ns.f_j = '- mu (uexact_i,j + uexact_j,i)_,i + pexact_,j'
            
        ### Primal problem ###
        ns.ubasis, ns.pbasis, ns.lbasis= function.chain([domain.basis('th-spline', degree=degree, continuity=degree-1).vector(2),
                                               domain.basis('th-spline', degree=degree-1, continuity=degree-2),
                                               [1],])

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

        # Shear traction
        ns.t_j = 'stress_ij n_i -  (stress_il n_i) n_l n_j'
       
        # boundary condition
        cons = domain.boundary.project(0, onto=ns.ubasis, geometry=geom, degree=degree*2)
    
        res = domain.integral('(f_i v_i - stress_ij v_i,j + q (uh_l,l - lh) - ph l) d:x' @ ns, degree=degree*2)
        
        trail = solver.solve_linear('trail', res.derivative('test'), constrain=cons)

        arguments['trail'] = trail
    

        ### Dual problem ###
        dualdegree = degree + 1
    
        ns.zbasis, ns.sbasis, ns.Lbasis = function.chain([domain.basis('th-spline', degree=dualdegree, continuity=dualdegree-1).vector(2),
                                                          domain.basis('th-spline', degree=dualdegree-1, continuity=dualdegree-2),
                                                          [1],])
        
        # Trail functions
        ns.z_i = 'zbasis_ni ?dualtrail_n'
        ns.s   = 'sbasis_n ?dualtrail_n'
        ns.Lh  = 'Lbasis_n ?dualtrail_n'
        
        # Test functions
        ns.V_i = 'zbasis_ni ?dualtest_n'
        ns.Q   = 'sbasis_n ?dualtest_n'
        ns.L   = 'Lbasis_n ?dualtest_n'

        ns.T_j = '(mu (V_i,j + V_j,i) - Q δ_ij) n_i -  ((mu (V_i,l + V_l,i) - Q δ_il) n_i) n_l n_j'

        dualcons = domain.boundary.project(0, onto=ns.zbasis, geometry=geom, degree=dualdegree*2)
        
        res  = domain.boundary['right'].integral('(T_i n_i) d:x' @ ns, degree=dualdegree*2)
        res -= domain.integral('((z_j,i + z_i,j) V_j,i - s V_k,k + Q (z_l,l - Lh) + s L) d:x' @ ns, degree=dualdegree*2)
    
        dualtrail = solver.solve_linear('dualtrail', res.derivative('dualtest'), constrain=dualcons)

        arguments['dualtrail'] = dualtrail
        
        ns.Iz   = domain.projection(ns.z, ns.ubasis, geometry=geom, degree=degree*2, constrain=cons, arguments=arguments)
        ns.Is   = domain.projection(ns.s, ns.pbasis, geometry=geom, degree=degree*2, arguments=arguments)

        # indicators
        ns.moment = '(f_i + (uh_j,i + uh_i,j)_,j - ph_,i) (f_i + (uh_l,i + uh_i,l)_,l - ph_,i)'
        ns.zsharp = '(z_i - Iz_i) (z_i - Iz_i)'

        ns.incomp = '(uh_i,i)^2'
        ns.ssharp = '(s - Is)^2'

        h = np.sqrt(indicater.integrate(domain, geom, degree, 1, domain, arguments=arguments))
        moment = np.sqrt(indicater.integrate(domain, geom, degree, ns.moment, domain, arguments=arguments))
        incomp = np.sqrt(indicater.integrate(domain, geom, degree, ns.incomp, domain, arguments=arguments))
    
        z_int  = np.sqrt(indicater.integrate(domain, geom, degree, ns.zsharp, domain, arguments=arguments))
        s_int  = np.sqrt(indicater.integrate(domain, geom, degree, ns.ssharp, domain, arguments=arguments))

        if method == 'goaloriented':
            
            indicators = incomp*s_int + moment*z_int 
            plotter.plot_indicators(method+'contributions_'+str(nref), domain, geom, {'Momentum':moment,'z_sharp':z_int,'incompressibility':incomp,'s_sharp':s_int}, normalize=False)
            plotter.plot_indicators(method+'indicators_'+str(nref), domain, geom, {'indicator':indicators,'incomp':incomp*s_int,'momentum':moment*z_int}, normalize=False)
            domain, refined = refiner.refine(domain, indicators, num, evalbasis, maxlevel=maxreflevel+uref, marker_type=None, select_type=None, refined_check=True)
            plotter.plot_mesh(method+'mesh'+str(nref), domain, geom)

        if method == 'residualbased':

            indicators = incomp*h + moment*h 
            plotter.plot_indicators(method+'indicators_'+str(nref), domain, geom, {'indicator':indicators,'incomp':incomp*h,'momentum':moment*h}, normalize=False)
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

    
        #Pavg = domain.integrate('p d:x' @ns, ischeme='gauss5')
        #Phavg = domain.integrate('ph d:x' @ns, ischeme='gauss5')
        #print(Pavg)
        #print(Phavg)
        #
        #plotter.plot_solution('u0', domain, geom, ns.u0)
        #plotter.plot_solution('u1', domain, geom, ns.u1)

   #plotter.plot_mesh('mesh', domain, geom)
   #plotter.plot_solution('exact_pressure', domain, geom, ns.p)
   #plotter.plot_solution('approx_pressure', domain, geom, ns.ph)
   #plotter.plot_streamlines('exact_velocity', domain, geom, ns, ns.u)
   #plotter.plot_streamlines('approx_velocity', domain, geom, ns, ns.uh)
   #plotter.plot_streamlines('traction', domain, geom, ns, ns.t)
  
    plotter.plot_solution('dual_pressure', domain, geom, ns.s)
    plotter.plot_streamlines('dual_velocity', domain, geom, ns, ns.z)

with config(verbose=3,nprocs=6):
    cli.run(main)
