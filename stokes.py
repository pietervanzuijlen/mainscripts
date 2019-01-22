'''
In this code different forms of adaptivity are tested on an lshaped domain on which a laplace problem is solved. The exact solution to this problem is known. Goal-oriented, residual-based and uniform refinement will be compared.

'''
from nutils import *
from utilities import *
import numpy as np

def main(degree      = 3,
         maxref      = 3,
         npoints     = 5,
         num         = 0.75,
         uref        = 1,
         maxreflevel = 4,
         maxuref     = 4,
         M1          = 0.5):

    methods = ['goaloriented','residualbased']
    methods = ['goaloriented']
    methods = ['uniform']

    nelems    = {} 
    ndofs     = {} 
    error_est = {} 
    error_qoi = {} 

    for method in methods:

        nelems[method]    = []
        ndofs[method]     = []
        error_est[method] = []
        error_qoi[method] = []
        
        domain, geom = domainmaker.porous(uref=uref, M1=M1)
        ns = function.Namespace()

        ns.mu   = 1
        ns.pbar = 1
        ns.x    = geom

        for nref in range(maxref):
            
            ### Primal problem ###
            ns.ubasis, ns.pbasis = function.chain([domain.basis('th-spline', degree=degree, patchcontinuous=True, continuity=degree-1).vector(2),
                                                   domain.basis('th-spline', degree=degree-1, patchcontinuous=True, continuity=degree-2)])

            # Trail functions
            ns.u_i = 'ubasis_ni ?trail_n'
            ns.p = 'pbasis_n ?trail_n'
            
            # Test functions
            ns.v_i = 'ubasis_ni ?test_n'
            ns.q = 'pbasis_n ?test_n'
    
            # Stress
            ns.stress_ij = 'mu (u_i,j + u_j,i) - p δ_ij'

            # Inflow pressure
            ns.g_n = 'pbar n_n'
        
            # boundary condition
            sqr = domain.boundary['top,bottom,corners,circle'].integral('u_k u_k d:x' @ ns, degree=degree*2)
            cons = solver.optimize('trail', sqr, droptol=1e-15)
    
            res = domain.integral('(stress_ij v_i,j - q u_l,l) d:x' @ ns, degree=degree*2)
            res += domain.boundary['left'].integral('(g_i v_i) d:x' @ ns, degree=degree*2)
            
            trail = solver.solve_linear('trail', res.derivative('test'), constrain=cons)
            
            ns = ns(trail=trail) 
            ### Primal problem ###
    
    
            ### Dual problem ###
            dualdegree = degree + 1

            ns.dualubasis, ns.dualpbasis = function.chain([domain.basis('th-spline', degree=dualdegree, patchcontinuous=True, continuity=dualdegree-1).vector(2),
                                                           domain.basis('th-spline', degree=dualdegree-1, patchcontinuous=True, continuity=dualdegree-2)])
    
            # Trail functions
            ns.z_i = 'dualubasis_ni ?dualtrail_n'
            ns.s = 'dualpbasis_n ?dualtrail_n'
    
            # Test functions
            ns.v_i = 'dualubasis_ni ?dualtest_n'
            ns.q = 'dualpbasis_n ?dualtest_n'
    
            sqr = domain.boundary['top,bottom,corners,circle'].integral('z_k z_k d:x' @ ns, degree=dualdegree*2)
            cons = solver.optimize('dualtrail', sqr, droptol=1e-15)
    
            res = domain.integral('(( mu (v_i,j + v_j,i) - q δ_ij ) z_i,j - s v_i,i ) d:x' @ ns, degree=dualdegree*2).derivative('dualtest')

            # Quantity of interest: outflow
            res += domain.boundary['left'].integral('( -v_i n_i ) d:x' @ ns, degree=dualdegree*2).derivative('dualtest')

            dualtrail = solver.solve_linear('dualtrail', res, constrain=cons)
    
            ns = ns(dualtrail=dualtrail)
    
            ns.Iz   = domain.project(ns.z, ns.ubasis, geometry=geom, ischeme='gauss4') 
            ns.Iz_i = 'ubasis_ni Iz_n'
            ns.Is   = domain.project(ns.s, ns.pbasis, geometry=geom, ischeme='gauss4') 
            ns.Is   = 'pbasis_n Is_n'
            ### Dual problem ###
    
            
            ### Get errors ###
            nelems[method]    += [len(domain)]
            ndofs[method]     += [len(ns.ubasis)]
            error_est[method] += [domain.integrate(function.norm2('(-mu (u_i,jj + u_j,ij) + p_,i) d:x' @ns)+function.abs('u_i,i d:x' @ns), ischeme='gauss5')]
            error_qoi[method] += [domain.integrate('(-mu (u_i,j + u_j,i) (z_i,j - Iz_i,j) + p (z_i,i - Iz_i,i) + s u_i,i) d:x' @ns, ischeme='gauss5')+domain.boundary['left'].integrate('g_i (z_i - Iz_i) d:x' @ns, ischeme='gauss5')]
            ### Get errors ###
    
    
            ### Get indicators ###
            ns.inflow  = '(g_i + stress_ij n_j) (g_i + stress_ik n_k)'
            ns.outflow = '(- stress_ij n_j) (- stress_ik n_k)'
            ns.jump    = '([-stress_ij] n_j) ([-stress_ik] n_k)'
            ns.force   = '(stress_ij,j) (stress_ik,k)'
            ns.incom   = '(u_i,i)^2'
            ns.zsharp  = '(z_i - Iz_i) (z_i - Iz_i)'
            ns.ssharp  = '(s - Is)^2'

            h = np.sqrt(indicater.integrate(domain, geom, degree, 1, domain))

            incom    = np.sqrt(indicater.integrate(domain, geom, degree, ns.incom, domain))
            force    = np.sqrt(indicater.integrate(domain, geom, degree, ns.force, domain))
            jump     = np.sqrt(indicater.integrate(domain, geom, degree, ns.jump, domain.interfaces, interfaces=True))
            inflow   = np.sqrt(indicater.integrate(domain, geom, degree, ns.inflow, domain.boundary['left']))
            outflow  = np.sqrt(indicater.integrate(domain, geom, degree, ns.outflow, domain.boundary['right']))
    
            z_int     = np.sqrt(indicater.integrate(domain, geom, degree, ns.zsharp, domain))
            s_int     = np.sqrt(indicater.integrate(domain, geom, degree, ns.ssharp, domain))
            z_jump    = np.sqrt(indicater.integrate(domain, geom, degree, ns.zsharp, domain.interfaces, interfaces=True))
            z_inflow  = np.sqrt(indicater.integrate(domain, geom, degree, ns.zsharp, domain.boundary['left']))
            z_outflow = np.sqrt(indicater.integrate(domain, geom, degree, ns.zsharp, domain.boundary['right']))

            rz = z_int + z_jump + z_inflow + z_outflow
            ### Get indicaters ###

            #plotter.plot_streamlines('velocity'+str(nref), domain, geom, ns, ns.u) 
            #plotter.plot_streamlines('dual_velocity'+str(nref), domain, geom, ns, ns.z) 
            #plotter.plot_solution('pressure'+str(nref), domain, geom, ns.p) 
            plotter.plot_solution('force'+str(nref), domain, geom, ns.force) 
            #plotter.plot_solution('incomp'+str(nref), domain, geom, ns.incom) 
    
    
            ### Refine mesh ###
            if method == 'goaloriented':

                # assemble indicaters
                inter = incom*s_int + force*z_int
                jump  = jump*z_jump
                bound = inflow*z_inflow + outflow*z_outflow

                indicators =  inter + jump + bound 

                plotter.plot_indicators('residual_contributions_'+str(nref), domain, geom, {'force':force,'incompressibility':incom,'interfaces':jump,'boundaries':inflow+outflow}, normalize=False)
                plotter.plot_indicators('sharp_contributions_'+str(nref), domain, geom, {'z_internal':z_int,'s_internal':s_int,'z_interfaces':z_jump,'z_boundaries':z_inflow+outflow}, normalize=False)
                plotter.plot_indicators('indicators_'+method+'_'+str(nref), domain, geom, {'indicator':indicators,'internal':inter,'interfaces':jump,'boundaries':bound}, normalize=False)

                domain = refiner.fractional_marking(domain, indicators, num, ns.pbasis, maxlevel=maxreflevel+uref)
                plotter.plot_levels('mesh_'+str(nref), domain, geom)

            if method == 'residualbased':

                # assemble indicaters
                inter = (incom + force)*h
                jump  = jump*np.sqrt(h)
                bound = (inflow + outflow)*np.sqrt(h)

                indicators =  inter + jump + bound 

                plotter.plot_indicators('indicators_'+method+'_'+str(nref), domain, geom, {'indicator':indicators})
                
                domain = refiner.fractional_marking(domain, indicators, num, ns.pbasis, maxlevel=maxreflevel+uref)
                plotter.plot_mesh('mesh_'+str(nref), domain, geom)

            if method == 'uniform':

                domain = domain.refine(1)

                if nref == maxuref:
                    break
    
    plotter.plot_convergence('Estimated_error',ndofs,error_est,labels=['dofs','Estimated error'])
    plotter.plot_convergence('Estimated_error_in_QoI',ndofs,error_qoi,labels=['dofs','Estimated error in QoI'])
    plotter.plot_convergence('Dofs_vs_elems',nelems,ndofs,labels=['nelems','ndofs'])

    anouncer.drum()

with config(verbose=3,nprocs=6):
    cli.run(main)
