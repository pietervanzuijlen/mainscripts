'''
LAPLACE: In this code different forms of adaptivity are tested on an lshaped domain on which a laplace problem is solved. The exact solution to this problem is known. Goal-oriented, residual-based and uniform refinement will be tested and compared. Utilities toolbox is used.  
''' 

from nutils import * 
import numpy as np
from utilities import *

def main(degree      = 2,
         refinements = 15,
         num         = 0.2,
         uref        = 2,
         maxreflevel = 7,
         maxuref     = 3):

    methods = ['goaloriented','residualbased','uniform']
    methods = ['goaloriented']

    nelems    = {} 
    elemsize  = {}
    ndofs     = {} 
    error_sol = {} 
    error_qoi = {} 

    for method in methods:

        nelems[method]    = []
        elemsize[method]  = []
        ndofs[method]     = []
        error_sol[method] = []
        error_qoi[method] = []
        
        domain, geom = domainmaker.lshape(uref=3, width=2, height=2)

        ns = function.Namespace()
        ns.x = geom
        ns.eps = np.mean(domain.integrate_elementwise(function.J(geom), degree=degree))/4
        ns.x0 = '(x_0 - 1)^2 + (x_1 + 1)^2 - eps^2'
        ns.x1 = function.min(ns.x0, 0)
        ns.k1 = 'exp( - x1 / eps^2 )'
        ns.C = 1/domain.integrate('k1 d:x' @ns, degree=degree*2)
        ns.k2 = 'C exp( - x1 / eps^2 )'

        for nref in range(refinements):
            log.user(method+': '+str(nref))
            
            ### Primal problem ###
            ns.basis = domain.basis('th-spline', degree=degree, patchcontinuous=True, continuity=degree-1)

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
        
            A    = domain.integrate(ns.eval_ij('basis_i,k basis_j,k d:x'), degree=degree*2)
            b    = domain.boundary['left'].integrate(ns.eval_i('basis_i g1 d:x'), degree=degree*2)
            b   += domain.boundary['top'].integrate(ns.eval_i('basis_i g2 d:x'), degree=degree*2)
            b   += domain.boundary['right'].integrate(ns.eval_i('basis_i g3 d:x'), degree=degree*2)
            b   += domain.boundary['bottom'].integrate(ns.eval_i('basis_i g4 d:x'), degree=degree*2)
    
            cons = domain.boundary['inner'].project(0, onto=ns.basis, geometry=geom, degree=degree*2)
        
            lhs = A.solve(b, constrain=cons)
            ns = ns(lhs=lhs)
            ### Primal problem ###
    
    
            ### Dual problem ###
            dualdegree = degree + 1

            ns.dualbasis = domain.basis('th-spline', degree=dualdegree, patchcontinuous=True, continuity=degree)
    
            ns.z = 'dualbasis_n ?duallhs_n'
        
            B = domain.integrate(ns.eval_ij('dualbasis_i,k dualbasis_j,k d:x'), degree=dualdegree*2)


            # single patch
            #Q = domain.boundary['bottom'].boundary['top'].integrate(ns.eval_i('dualbasis_i d:x'), degree=dualdegree*2)
            # multi patch
            Q = domain.boundary['bottom'].boundary['right'].integrate(ns.eval_i('dualbasis_i d:x'), degree=dualdegree*2)
            # mollification
            #Q = domain.integrate(ns.eval_i('k2 dualbasis_i d:x'), degree=dualdegree*2)

    
            consdual = domain.boundary['inner'].project(0, onto=ns.dualbasis, geometry=geom, degree=dualdegree*2)
        
            duallhs = B.solve(Q, constrain = consdual)
            ns      = ns(duallhs=duallhs)
        
            ns.Iz   = domain.projection(ns.z, ns.basis, geometry=geom, degree=dualdegree*2, constrain=cons)
            ### Dual problem ###

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
            rbound1 = np.sqrt(indicater.integrate(domain, geom, degree, ns.rbound1, domain.boundary['left']))
            rbound2 = np.sqrt(indicater.integrate(domain, geom, degree, ns.rbound2, domain.boundary['top']))
            rbound3 = np.sqrt(indicater.integrate(domain, geom, degree, ns.rbound3, domain.boundary['right']))
            rbound4 = np.sqrt(indicater.integrate(domain, geom, degree, ns.rbound4, domain.boundary['bottom']))
    
            rz_int    = np.sqrt(indicater.integrate(domain, geom, degree, ns.rz, domain))
            rz_jump   = np.sqrt(indicater.integrate(domain, geom, degree, ns.rz, domain.interfaces, interfaces=True))
            rz_bound1 = np.sqrt(indicater.integrate(domain, geom, degree, ns.rz, domain.boundary['left']))
            rz_bound2 = np.sqrt(indicater.integrate(domain, geom, degree, ns.rz, domain.boundary['top']))
            rz_bound3 = np.sqrt(indicater.integrate(domain, geom, degree, ns.rz, domain.boundary['right']))
            rz_bound4 = np.sqrt(indicater.integrate(domain, geom, degree, ns.rz, domain.boundary['bottom']))

            rz = rz_int + rz_jump + rz_bound1 + rz_bound2 + rz_bound3 + rz_bound4
            ### Get indicaters ###

            ### Get errors ###
            nelems[method]    += [len(domain)]
            elemsize[method]  += [1/min(h)]
            ndofs[method]     += [len(ns.basis)]
            error_sol[method] += [np.sqrt(domain.integrate('(u - uh)^2 d:x' @ns, degree=10))]
            error_qoi[method] += [np.sqrt(domain.boundary['bottom'].boundary['right'].integrate('(u - uh)^2 d:x' @ ns , ischeme='gauss1'))]
            ### Get errors ###

            ### Refine mesh ###
            if method == 'goaloriented':

                # assemble indicaters
                inter = rint*rz_int
                jump = rjump*rz_jump
                bound = rbound1*rz_bound1 + rbound2*rz_bound2 + rbound3*rz_bound3 + rbound4*rz_bound4

                indicators =  inter + jump + bound 

                plotter.plot_indicators('indicators_'+method+'_'+str(nref), domain, geom, {'indicator':indicators,'internal':inter,'interfaces':jump,'boundary':bound}, alpha=.5)
                domain, refined = refiner.refine(domain, indicators, num, ns.basis, maxlevel=maxreflevel+uref+1, marker_type=None, select_type=None)

            if method == 'residualbased':

                # assemble indicaters
                inter = rint*h
                jump = rjump*np.sqrt(h)
                bound = (rbound1 + rbound2 + rbound3 + rbound4)*np.sqrt(h)

                indicators =  inter + jump + bound 

                plotter.plot_indicators('indicators_'+method+'_'+str(nref), domain, geom, {'indicator':indicators,'internal':inter,'interfaces':jump,'boundary':bound}, alpha=.5)
                domain, refined = refiner.refine(domain, indicators, num, ns.basis, maxlevel=maxreflevel+uref+1, marker_type=None, select_type=None)

            if method == 'uniform':

                domain = domain.refine(1)
                refined = True

                if nref == maxuref:
                    break

            if not refined:
                break
            ### Refine mesh ###

        ### Postprocessing ###
        plotter.plot_mesh(method+'mesh', domain, geom)
        plotter.plot_solution(method+'dualsolution', domain, geom, ns.z)
        plotter.plot_solution(method+'solution', domain, geom, ns.u)
    
        writer.write('../results/laplace/'+method+'mollification', {'degree':degree, 'uref':uref, 'maxuref':maxuref, 'refinements':refinements, 'num':num},
                     ndofs=ndofs, nelems=nelems, error_sol=error_sol, error_qoi=error_qoi)

    plotter.plot_convergence('Exact_error',ndofs,error_sol,labels=['dofs','Exact error'],slopemarker=True)
    plotter.plot_convergence('Exact_error_elemsize',elemsize,error_sol,labels=['1/h','Exact error'])
    plotter.plot_convergence('Error_in_QoI',ndofs,error_qoi,labels=['dofs','Error in QoI'],slopemarker=True)
    plotter.plot_convergence('Dofs_vs_elems',nelems,ndofs,labels=['nelems','ndofs'])
        ### Postprocessing ###

    #anouncer.drum()

with config(verbose=3,nprocs=6):
    cli.run(main)
