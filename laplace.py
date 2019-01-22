'''
In this code different forms of adaptivity are tested on an lshaped domain on which a laplace problem is solved. The exact solution to this problem is known. Goal-oriented, residual-based and uniform refinement will be compared.

'''
from   nutils import *
import numpy as np
from utilities import *

def main(degree      = 2,
         maxref      = 6,
         npoints     = 5,
         num         = 0.5,
         uref        = 2,
         maxreflevel = 8,
         maxuref     = 4):

    methods = ['goaloriented','uniform','residualbased']
    methods = ['goaloriented']

    nelems    = {} 
    ndofs     = {} 
    error_sol = {} 
    error_qoi = {} 

    for method in methods:

        nelems[method]    = []
        ndofs[method]     = []
        error_sol[method] = []
        error_qoi[method] = []
        
        domain, geom = domainmaker.lshape(uref=uref, width=2, height=2)
        ns = function.Namespace()

        for nref in range(maxref):
            log.user(method+': '+str(nref))
            
            ### Primal problem ###
            ns.basis = domain.basis('th-spline', degree=degree, patchcontinuous=True, continuity=degree-1)
            ns.x = geom
            x, y = geom
            th = function.ArcTan2(y,x) 
            R  = (x**2 + y**2)**.5
    
            #neumann BC
            ns.g1 = -2 * (x*function.sin((2*th+np.pi)/3) - y*function.cos((2*th+np.pi)/3))/(3*(x**2+y**2)**(2/3))
            ns.g2 =  2 * (y*function.sin((2*th+np.pi)/3) + x*function.cos((2*th+np.pi)/3))/(3*(x**2+y**2)**(2/3))
            ns.g3 =  2 * (x*function.sin((2*th+np.pi)/3) - y*function.cos((2*th+np.pi)/3))/(3*(x**2+y**2)**(2/3))
            ns.g4 = -2 * (y*function.sin((2*th+np.pi)/3) + x*function.cos((2*th+np.pi)/3))/(3*(x**2+y**2)**(2/3))
    
            #exact solution
            ns.u  = R**(2/3) * function.Sin(2/3*th+np.pi/3) 
    
            ns.uh = 'basis_n ?lhs_n'
            ns.e  = 'u - uh'
        
            A    = domain.integrate(ns.eval_ij('basis_i,k basis_j,k d:x'), degree=degree*2)
            b    = domain.boundary['patch1-top'].integrate(ns.eval_i('basis_i g1 d:x'), degree=degree*2)
            b   += domain.boundary['patch1-right'].integrate(ns.eval_i('basis_i g2 d:x'), degree=degree*2)
            b   += domain.boundary['patch0-right'].integrate(ns.eval_i('basis_i g3 d:x'), degree=degree*2)
            b   += domain.boundary['patch0-bottom'].integrate(ns.eval_i('basis_i g4 d:x'), degree=degree*2)
    
            cons = domain.boundary['patch0-left,patch1-left'].project(0, onto=ns.basis, geometry=geom, degree=degree*2)
        
            lhs = A.solve(b, constrain=cons)
            ns = ns(lhs=lhs)
            ### Primal problem ###
    
    
            ### Dual problem ###
            dualdegree = degree + 1
            ns.dualbasis = domain.basis('th-spline', degree=dualdegree, patchcontinuous=True, continuity=degree)
    
            ns.z = 'dualbasis_n ?duallhs_n'
        
            B = domain.integrate(ns.eval_ij('dualbasis_i,k dualbasis_j,k d:x'), degree=dualdegree*2)
            Q = domain.boundary['patch0-bottom'].boundary['right'].integrate(ns.eval_i('dualbasis_i d:x'), degree=dualdegree*2)
    
            consdual = domain.boundary['patch0-left,patch1-left'].project(0, onto=ns.dualbasis, geometry=geom, degree=dualdegree*2)
        
            duallhs = B.solve(Q, constrain = consdual)
            ns      = ns(duallhs=duallhs)
        
            ns.Iz   = domain.projection(ns.z, ns.basis, geometry=geom, degree=dualdegree*2, constrain=cons)
            ### Dual problem ###
    
            
            ### Get errors ###
            nelems[method]    += [len(domain)]
            ndofs[method]     += [len(ns.basis)]
            error_sol[method] += [abs(domain.integrate('(u - uh) d:x' @ns, degree=degree*2))]
            error_qoi[method] += [abs(domain.boundary['patch0-bottom'].boundary['right'].integrate('(u - uh) d:x' @ ns , ischeme='gauss1'))]
            ### Get errors ###
    

            ### Get indicaters ###
            ns.rint    = '(uh_,ii)^2'
            ns.rjump   = '(.5 [uh_,n] n_n)^2'
            ns.rbound1 = '(g1 - uh_,n n_n)^2'
            ns.rbound2 = '(g2 - uh_,n n_n)^2'
            ns.rbound3 = '(g3 - uh_,n n_n)^2'
            ns.rbound4 = '(g4 - uh_,n n_n)^2'
            ns.rz      = '(z - Iz)^2'
    
            h = np.sqrt(indicater.integrate(domain, geom, degree, 1, domain))

            rint    = np.sqrt(indicater.integrate(domain, geom, degree, ns.rint, domain))
            rjump   = np.sqrt(indicater.integrate(domain, geom, degree, ns.rjump, domain.interfaces, interfaces=True))
            rbound1 = np.sqrt(indicater.integrate(domain, geom, degree, ns.rbound1, domain.boundary['patch1-top']))
            rbound2 = np.sqrt(indicater.integrate(domain, geom, degree, ns.rbound2, domain.boundary['patch1-right']))
            rbound3 = np.sqrt(indicater.integrate(domain, geom, degree, ns.rbound3, domain.boundary['patch0-right']))
            rbound4 = np.sqrt(indicater.integrate(domain, geom, degree, ns.rbound4, domain.boundary['patch0-bottom']))
    
            rz_int    = np.sqrt(indicater.integrate(domain, geom, degree, ns.rz, domain))
            rz_jump   = np.sqrt(indicater.integrate(domain, geom, degree, ns.rz, domain.interfaces, interfaces=True))
            rz_bound1 = np.sqrt(indicater.integrate(domain, geom, degree, ns.rz, domain.boundary['patch1-top']))
            rz_bound2 = np.sqrt(indicater.integrate(domain, geom, degree, ns.rz, domain.boundary['patch1-right']))
            rz_bound3 = np.sqrt(indicater.integrate(domain, geom, degree, ns.rz, domain.boundary['patch0-right']))
            rz_bound4 = np.sqrt(indicater.integrate(domain, geom, degree, ns.rz, domain.boundary['patch0-bottom']))

            rz = rz_int + rz_jump + rz_bound1 + rz_bound2 + rz_bound3 + rz_bound4
            ### Get indicaters ###
    
    
            ### Refine mesh ###
            if method == 'goaloriented':

                # assemble indicaters
                inter = rint*rz_int
                jump = rjump*rz_jump
                bound = rbound1*rz_bound1 + rbound2*rz_bound2 + rbound3*rz_bound3 + rbound4*rz_bound4

                indicators =  inter + jump + bound 

                #plotter.plot_indicators('residual_contributions_'+str(nref), domain, geom, {'internal':rint,'interfaces':rjump,'boundary':rbound1+rbound2+rbound3+rbound4})
                #plotter.plot_indicators('sharp_contributions_'+str(nref), domain, geom, {'internal':rz_int,'interfaces':rz_jump,'boundary':rz_bound1+rz_bound2+rz_bound3+rz_bound4})
                plotter.plot_indicators('indicators_'+method+'_'+str(nref), domain, geom, {'indicator':indicators,'internal':inter,'interfaces':jump,'boundary':bound})
                plotter.plot_mesh('mesh_'+str(nref), domain, geom)

                domain = refiner.dorfler_marking(domain, indicators, num, ns.basis, maxlevel=maxreflevel+uref, select_type='supp_only')

            if method == 'residualbased':

                # assemble indicaters
                inter = rint*h
                jump = rjump*np.sqrt(h)
                bound = (rbound1 + rbound2 + rbound3 + rbound4)*np.sqrt(h)

                indicators =  inter + jump + bound 

                #plotter.plot_indicators('indicators_'+method+'_'+str(nref), domain, geom, {'indicator':indicators})
                #plotter.plot_mesh('mesh_'+str(nref), domain, geom)
                domain = refiner.dorfler_marking(domain, indicators, num, ns.basis, maxlevel=maxreflevel+uref, select_type='same_level')

            if method == 'uniform':

                domain = domain.refine(1)

                if nref == maxuref:
                    break
            ### Refine mesh ###
        #plotter.plot_levels('mesh_'+method, domain, geom)
    
    plotter.plot_convergence('Exact_error',ndofs,error_sol,labels=['dofs','Exact error'],slopemarker=True)
    #plotter.plot_convergence('Error_in_QoI',ndofs,error_qoi,labels=['dofs','Error in QoI'])
    #plotter.plot_convergence('Dofs_vs_elems',nelems,ndofs,labels=['nelems','ndofs'])

    #anouncer.drum()

with config(verbose=3,nprocs=6):
    cli.run(main)
