'''
In this code different forms of adaptivity are tested on an lshaped domain on which a laplace problem is solved. The exact solution to this problem is known. Goal-oriented, residual-based and uniform refinement will be compared.

'''
from nutils import *
from utilities import *
import numpy as np

def main(degree      = 3,
         maxref      = 10,
         npoints     = 5,
         num         = 0.5,
         uref        = 1,
         maxreflevel = 5,
         maxuref     = 3,
         M1          = 0.5):

    methods = ['goaloriented','residualbased','uniform']
    methods = ['uniform']

    nelems    = {} 
    ndofs     = {} 
    error_force = {} 
    error_incomp = {} 
    error_qoi = {} 
    error_zincomp = {} 

    for method in methods:

        nelems[method]       = []
        ndofs[method]        = []
        error_force[method]  = []
        error_incomp[method] = []
        error_qoi[method] = []
        error_zincomp[method] = []
        
        domain, geom = domainmaker.porous(uref=uref, M1=M1)
        ns = function.Namespace()

        ns.x    = geom

        for nref in range(maxref):
            
            ### Primal problem ###
            ns.ubasis, ns.pbasis = function.chain([domain.basis('th-spline', degree=degree, patchcontinuous=True, continuity=degree-1).vector(2),
                                                   domain.basis('th-spline', degree=degree-1, patchcontinuous=True, continuity=degree-2)])

            # Search for better solution !!
            evalbasis = domain.basis('th-spline', degree=degree)

            # Trail functions
            ns.u_i = 'ubasis_ni ?trail_n'
            ns.p = 'pbasis_n ?trail_n'
            
            # Test functions
            ns.v_i = 'ubasis_ni ?test_n'
            ns.q = 'pbasis_n ?test_n'
    
            # Inflow traction 
            ns.g_n = '-n_n'
        
            # boundary condition
            cons = domain.boundary['top,bottom,corners,circle'].project(0, onto=ns.ubasis, geometry=geom, degree=degree*2)
    
            res = domain.boundary['left'].integral('(g_i v_i) d:x' @ ns, degree=degree*2)
            res -= domain.integral('(u_j,i v_j,i - p v_k,k + q u_l,l) d:x' @ ns, degree=degree*2)
            
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
    
            consdual = domain.boundary['top,bottom,corners,circle'].project(0, onto=ns.dualubasis, geometry=geom, degree=dualdegree*2)
    
            res = domain.boundary['right'].integral('(n_i v_i) d:x' @ ns, degree=dualdegree*2)
            res -= domain.integral('((z_j,i + z_i,j) v_j,i - s v_k,k + q z_l,l) d:x' @ ns, degree=dualdegree*2)

            dualtrail = solver.solve_linear('dualtrail', res.derivative('dualtest'), constrain=consdual)
    
            ns = ns(dualtrail=dualtrail)

            ns.Iz   = domain.projection(ns.z, ns.ubasis, geometry=geom, degree=degree*2, constrain=cons)
            ns.Is   = domain.projection(ns.s, ns.pbasis, geometry=geom, degree=degree*2)
            ### Dual problem ###
    
            
            ### Get errors ###
            nelems[method]    += [len(domain)]
            ndofs[method]     += [len(ns.ubasis)]

            
            Rz  = domain.boundary['left'].integrate('g_i  z_i d:x' @ns, ischeme='gauss5') - domain.integrate('(u_i,j  z_i,j - p  z_k,k) d:x' @ns, ischeme='gauss5')
            RIz = domain.boundary['left'].integrate('g_i Iz_i d:x' @ns, ischeme='gauss5') - domain.integrate('(u_i,j Iz_i,j - p Iz_k,k) d:x' @ns, ischeme='gauss5')
            Rz_Iz = domain.boundary['left'].integrate('g_i (z_i - Iz_i) d:x' @ns, ischeme='gauss5') - domain.integrate('((u_j,i + u_i,j)  (z_i,j - Iz_i,j) - p (z_k,k - Iz_k,k)) d:x' @ns, ischeme='gauss5')

            Rs = domain.integrate('(s u_i,i) d:x' @ns, ischeme='gauss5')
            RIs = domain.integrate('(Is u_i,i) d:x' @ns, ischeme='gauss5')

            error_force[method]  += [domain.integrate(function.norm2('(-(u_j,i)_,i + p_,j) d:x' @ns), ischeme='gauss5')]
            error_incomp[method] += [domain.integrate(function.abs('u_i,i d:x' @ns), ischeme='gauss5')]
            error_qoi[method]    += [abs(Rz)]
            error_zincomp[method] += [abs(Rs)]

            print('R(z): ', Rz)
            print('R(Iz): ', RIz)
            print('R(z-Iz): ', Rz_Iz)

            print('R(s): ', Rs)
            print('R(Is): ', RIs)
            
            print('velocity laplacian :', domain.integrate(function.norm2('u_j,ii d:x' @ns), ischeme='gauss5'))
            print('pressure gradient :',domain.integrate(function.norm2('p_,j d:x' @ns), ischeme='gauss5'))
            print('mean pressure :',domain.integrate('p d:x' @ns, ischeme='gauss5'))

            ### Get errors ###
    
    
            ### Get indicators ###

            ns.inflow  = '(g_i - (u_j,i + u_i,j) n_j + p n_i) (g_i - (u_l,i + u_i,l) n_l + p n_i)'
            ns.outflow = '(-(u_j,i + u_i,j) n_j + p n_i) (-(u_l,i + u_i,l) n_l + p n_i)'
            ns.jump    = '([-(u_j,i + u_i,j)] n_j + [p] n_i) ([-(u_l,i + u_i,l)] n_l + [p] n_i)'
            ns.force   = '((u_j,i + u_i,j)_,j - p_,i) ((u_l,i + u_i,l)_,l - p_,i)'
            ns.zsharp  = '(z_i - Iz_i) (z_i - Iz_i)'

            ns.incom   = '(u_i,i)^2'
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

            plotter.plot_streamlines('velocity'+str(nref), domain, geom, ns, ns.u) 
            plotter.plot_solution('pressure'+str(nref), domain, geom, ns.p) 
            
            ns.momentum_i = '(u_j,i + u_i,j)_,j - p_,i'
            plotter.plot_streamlines('momentum'+str(nref), domain, geom, ns, ns.momentum) 
            #plotter.plot_solution('force'+str(nref), domain, geom, ns.force) 
            #plotter.plot_solution('incomp'+str(nref), domain, geom, ns.incom) 
    
    
            ### Refine mesh ###
            if method == 'goaloriented':

                # assemble indicaters
                inter = incom*s_int + force*z_int
                iface = jump*z_jump
                bound = inflow*z_inflow + outflow*z_outflow

                indicators =  inter + iface + bound 

                plotter.plot_indicators('residual_contributions_'+str(nref), domain, geom, {'force':force,'incompressibility':incom,'interfaces':jump,'boundaries':inflow+outflow}, normalize=False)
                plotter.plot_indicators('sharp_contributions_'+str(nref), domain, geom, {'z_internal':z_int,'s_internal':s_int,'z_interfaces':z_jump,'z_boundaries':z_inflow+z_outflow}, normalize=False)
                plotter.plot_indicators('indicators_'+method+'_'+str(nref), domain, geom, {'indicator':indicators,'internal':inter,'interfaces':iface,'boundaries':bound}, normalize=False)

                domain, refined = refiner.refine(domain, indicators, num, evalbasis, maxlevel=maxreflevel+uref, marker_type=None, select_type=None, refined_check=True)

            if method == 'residualbased':

                # assemble indicaters
                inter = (incom + force)*h
                iface = jump*np.sqrt(h)
                bound = (inflow + outflow)*np.sqrt(h)

                indicators =  inter + iface + bound 

                plotter.plot_indicators('residual_contributions_'+str(nref), domain, geom, {'force':force*h,'incompressibility':incom*h,'interfaces':jump*np.sqrt(h),'boundaries':(inflow+outflow)*np.sqrt(h)}, normalize=False)
                plotter.plot_indicators('indicators_'+method+'_'+str(nref), domain, geom, {'indicator':indicators})
                
                domain, refined = refiner.refine(domain, indicators, num, evalbasis, maxlevel=maxreflevel+uref, marker_type=None, select_type=None, refined_check=True)

            if method == 'uniform':

                domain = domain.refine(1)
                refined = True

                if nref == maxuref:
                    refined = False                    

            plotter.plot_levels(method+'mesh_'+str(nref), domain, geom, minlvl=uref)
            if not refined:
                break
    

    plotter.plot_convergence('Estimated_error_force',ndofs,error_force,labels=['dofs','Estimated error'])
    plotter.plot_convergence('Estimated_error_incomp',ndofs,error_incomp,labels=['dofs','Estimated error'])
    plotter.plot_convergence('Estimated_error_in_QoI',ndofs,error_qoi,labels=['dofs','Estimated error in QoI'])
    plotter.plot_convergence('Estimated_error_in_zincomp',ndofs,error_zincomp,labels=['dofs','Estimated error in QoI'])
    plotter.plot_convergence('Dofs_vs_elems',nelems,ndofs,labels=['nelems','ndofs'])

    anouncer.drum()

with config(verbose=3,nprocs=6):
    cli.run(main)
