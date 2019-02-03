from nutils import *
from utilities import *
import numpy as np

def main(degree      = 3,
         uref        = 1,
         refinements = 5,
         num         = 0.5,
         npoints     = 5,
         maxreflevel = 5,
         maxuref     = 3,
         beta        = 50,
         M1          = 0.5,
         rc          = 0.2,
         H           = 1,):

    methods = ['goaloriented','residualbased','uniform']
    methods = ['goaloriented']

    nelems    = {} 
    ndofs     = {} 
    error_force = {} 
    error_incomp = {} 
    error_qoi = {} 
    error_zincomp = {} 

    for method in methods:
      for r1 in [.2]:

        nelems[method]       = []
        ndofs[method]        = []
        error_force[method]  = []
        error_incomp[method] = []
        error_qoi[method] = []
        error_zincomp[method] = []
        
        domain, geom = domainmaker.porous(uref=uref, M1=M1, r1=r1)
        ns = function.Namespace()

        ns.mu = 1
        ns.x  = geom

        for nref in range(refinements):
            
            ######################
            ### Primal problem ###
            ######################

            ns.ubasis, ns.pbasis = function.chain([domain.basis('th-spline', degree=degree, patchcontinuous=True, continuity=degree-2).vector(2),
                                                   domain.basis('th-spline', degree=degree-1, patchcontinuous=True, continuity=degree-2)])

            # Search for better solution !!
            evalbasis = domain.basis('th-spline', degree=degree)

            # Trail functions
            ns.u_i = 'ubasis_ni ?trail_n'
            ns.p = 'pbasis_n ?trail_n'
            
            # Test functions
            ns.v_i = 'ubasis_ni ?test_n'
            ns.q = 'pbasis_n ?test_n'

            # Stress
            ns.stress_ij = 'mu (u_i,j + u_j,i) - p δ_ij'
            #ns.stress_ij = 'mu u_i,j - p δ_ij'
    
            # Inflow traction 
            ns.g_n = '-n_n'
            #ns.rc= rc
            #ns.G = 100
            #ns.H = H
            #ns.g_n = 'G (x_1 - rc) (H - rc - x_1) n_n' 
        
            # boundary condition

            # Nitsche values
            ns.beta = beta 
            areas = domain.integrate_elementwise(function.J(geom), degree=degree)
            ns.he = function.elemwise(domain.transforms, areas)
            ns.nitsche  = 'mu ( (u_i,j + u_j,i) n_i) v_j + mu ( (v_i,j + v_j,i) n_i ) u_j - mu (beta / he) v_i u_i - p v_i n_i - q u_i n_i'
    
            res = domain.boundary['left'].integral('(g_i v_i) d:x' @ ns, degree=degree*2)
            res += domain.integral('(-stress_ij v_i,j + q u_l,l) d:x' @ ns, degree=degree*2)
            res += domain.boundary['top,bottom,corners,circle'].integral('nitsche d:x' @ns, degree=degree*2)

            trail = solver.solve_linear('trail', res.derivative('test'))
            ns = ns(trail=trail) 

    
            ####################
            ### Dual problem ###
            ####################

            dualdegree = degree + 1

            ns.dualubasis, ns.dualpbasis = function.chain([domain.basis('th-spline', degree=dualdegree, patchcontinuous=True, continuity=dualdegree-2).vector(2),
                                                           domain.basis('th-spline', degree=dualdegree-1, patchcontinuous=True, continuity=dualdegree-2)])
    
            # Trail functions
            ns.z_i = 'dualubasis_ni ?dualtrail_n'
            ns.s = 'dualpbasis_n ?dualtrail_n'
    
            # Test functions
            ns.v_i = 'dualubasis_ni ?dualtest_n'
            ns.q = 'dualpbasis_n ?dualtest_n'
   
            # Nitsche values
            ns.beta = 5
            ns.nitsche  = 'mu ( (z_i,j + z_j,i) n_i) v_j + mu ( (v_i,j + v_j,i) n_i ) z_j - mu (beta / he) v_i z_i - s v_i n_i - q z_i n_i'
    
            res = domain.boundary['right'].integral('n_i v_i d:x' @ ns, degree=dualdegree*2)
            res += domain.integral('(- mu (z_j,i + z_i,j) v_j,i + s v_k,k + q z_l,l) d:x' @ ns, degree=dualdegree*2)
            res += domain.boundary['top,bottom,corners,circle'].integral('nitsche d:x' @ns, degree=degree*2)

            dualtrail = solver.solve_linear('dualtrail', res.derivative('dualtest'))
            ns = ns(dualtrail=dualtrail)

            cons = domain.boundary['top,bottom,corners,circle'].project(0, onto=ns.ubasis, geometry=geom, degree=degree*2)
            ns.Iz   = domain.projection(ns.z, ns.ubasis, geometry=geom, degree=degree*2, constrain=cons)
            ns.Is   = domain.projection(ns.s, ns.pbasis, geometry=geom, degree=degree*2)
            ### Dual problem ###
    
            
            ### Get errors ###
            nelems[method]    += [len(domain)]
            ndofs[method]     += [len(ns.ubasis)]
            error_force[method]  += [domain.integrate(function.norm2('stress_ij,i d:x' @ns), ischeme='gauss5')]
            error_incomp[method] += [domain.integrate(function.abs('u_i,i d:x' @ns), ischeme='gauss5')]

            #Rz  = domain.boundary['left'].integrate('g_i  z_i d:x' @ns, ischeme='gauss5') - domain.integrate('(u_i,j  z_i,j - p  z_k,k) d:x' @ns, ischeme='gauss5')
            #RIz = domain.boundary['left'].integrate('g_i Iz_i d:x' @ns, ischeme='gauss5') - domain.integrate('(u_i,j Iz_i,j - p Iz_k,k) d:x' @ns, ischeme='gauss5')
            #Rz_Iz = domain.boundary['left'].integrate('g_i (z_i - Iz_i) d:x' @ns, ischeme='gauss5') - domain.integrate('((u_j,i + u_i,j)  (z_i,j - Iz_i,j) - p (z_k,k - Iz_k,k)) d:x' @ns, ischeme='gauss5')

            #Rs = domain.integrate('(s u_i,i) d:x' @ns, ischeme='gauss5')
            #RIs = domain.integrate('(Is u_i,i) d:x' @ns, ischeme='gauss5')

            #error_qoi[method]    += [abs(Rz)]
            #error_zincomp[method] += [abs(Rs)]

            #print('R(z): ', Rz)
            #print('R(Iz): ', RIz)
            #print('R(z-Iz): ', Rz_Iz)

            #print('R(s): ', Rs)
            #print('R(Is): ', RIs)

            print('velocity laplacian :', domain.integrate(function.norm2('u_j,ii d:x' @ns), ischeme='gauss5'))
            print('pressure gradient :',domain.integrate(function.norm2('p_,j d:x' @ns), ischeme='gauss5'))
            print('mean pressure :',domain.integrate('p d:x' @ns, ischeme='gauss5'))

            ns.momentum_i = 'mu (u_i,j + u_j,i)_,j + p_,i'
            ns.force_i    = 'mu (u_i,j + u_j,i)_,j'
            ns.presgrad_i = 'p_,i'
            plotter.plot_streamlines('momentum',domain,geom,ns,ns.momentum)
            #plotter.plot_streamlines('force',domain,geom,ns,ns.force)
            #plotter.plot_streamlines('presgrad',domain,geom,ns,ns.presgrad)

            ### Get errors ###
    
    
            ### Get indicators ###

            #ns.inflow  = '(g_i - (u_j,i + u_i,j) n_j + p n_i) (g_i - (u_l,i + u_i,l) n_l + p n_i)'
            #ns.jump    = '([-(u_j,i + u_i,j)] n_j + [p] n_i) ([-(u_l,i + u_i,l)] n_l + [p] n_i)'
            #ns.outflow = '(-(u_j,i + u_i,j) n_j + p n_i) (-(u_l,i + u_i,l) n_l + p n_i)'
            #ns.force   = '((u_j,i + u_i,j)_,j - p_,i) ((u_l,i + u_i,l)_,l - p_,i)'

            ns.inflow  = '(g_i - stress_ij n_j) (g_i - stress_il n_l)'
            ns.outflow  = '(- stress_ij n_j) (- stress_il n_l)'
            ns.jump    = '[stress_ij] n_j [stress_il] n_l'
            ns.force   = 'stress_ij,j stress_il,l'
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

            ns.zsharp = 'z_i z_i'
            ns.sharp = 's^2'
            z_int     = np.sqrt(indicater.integrate(domain, geom, degree, ns.zsharp, domain))
            s_int     = np.sqrt(indicater.integrate(domain, geom, degree, ns.ssharp, domain))

            ### Get indicaters ###
    
    
            ### Refine mesh ###
            if method == 'goaloriented':

                # assemble indicaters
                #inter = incom*s_int + force*z_int
                #iface = jump*z_jump
                #bound = inflow*z_inflow + outflow*z_outflow

                inter = h*incom*s_int + h*force*z_int
                iface = np.sqrt(h)*jump*z_int
                bound = np.sqrt(h)*(inflow*z_int + outflow*z_int)

                indicators =  inter + iface + bound 

                #plotter.plot_indicators('residual_contributions_'+str(nref), domain, geom, {'force':force,'incompressibility':incom,'interfaces':jump,'boundaries':inflow+outflow}, normalize=False, alpha=.5)
                #plotter.plot_indicators('sharp_contributions_'+str(nref), domain, geom, {'z_internal':z_int,'s_internal':s_int,'z_interfaces':z_jump,'z_boundaries':z_inflow+z_outflow}, normalize=False, alpha=.5)
                plotter.plot_indicators('indicators_'+method+'_'+str(nref), domain, geom, {'indicator':indicators,'internal':inter,'interfaces':iface,'boundaries':bound}, normalize=False, alpha=.5)

                domain, refined = refiner.refine(domain, indicators, num, evalbasis, maxlevel=maxreflevel+uref, select_type='same_level')

            if method == 'residualbased':

                # assemble indicaters
                inter = (incom + force)*h
                iface = jump*np.sqrt(h)
                bound = (inflow + outflow)*np.sqrt(h)

                indicators =  inter + iface + bound 

                #plotter.plot_indicators('residual_contributions_'+str(nref), domain, geom, {'force':force*h,'incompressibility':incom*h,'interfaces':jump*np.sqrt(h),'boundaries':(inflow+outflow)*np.sqrt(h)}, normalize=False, alpha=.5)
                plotter.plot_indicators('indicators_'+method+'_'+str(nref), domain, geom, {'indicator':indicators}, alpha=.5)
                
                domain, refined = refiner.refine(domain, indicators, num, evalbasis, maxlevel=maxreflevel+uref, select_type='same_level')

            if method == 'uniform':

                domain = domain.refine(1)
                refined = True

                if nref == maxuref:
                    refined = False                    

            # Stop with the refinement loop if nothing is refined
            if not refined:
                break

        plotter.plot_mesh('mesh',domain,geom)
        plotter.plot_streamlines('velocity',domain,geom,ns,ns.u)
        plotter.plot_solution('pressure',domain,geom,ns.p)
    
    plotter.plot_convergence('Estimated_error_force',ndofs,error_force,labels=['dofs','Estimated error'],slopemarker=True)
    plotter.plot_convergence('Estimated_error_incomp',ndofs,error_incomp,labels=['dofs','Estimated error'],slopemarker=True)
    #plotter.plot_convergence('Estimated_error_in_QoI',ndofs,error_qoi,labels=['dofs','Estimated error in QoI'],slopemarker=True)
    #plotter.plot_convergence('Estimated_error_in_zincomp',ndofs,error_zincomp,labels=['dofs','Estimated error in QoI'],slopemarker=True)
    plotter.plot_convergence('Dofs_vs_elems',nelems,ndofs,labels=['nelems','ndofs'])

    anouncer.drum()

with config(verbose=3,nprocs=6):
    cli.run(main)
