'''
In this code different forms of adaptivity are tested on an lshaped domain on which a laplace problem is solved. The exact solution to this problem is known. Goal-oriented, residual-based and uniform refinement will be compared.

'''

from   nutils import *
import numpy as np
from utilities import plotter
from utilities import indicater 
from utilities import writer
from utilities import domainmaker
from utilities import refiner

import treelog

def main(degree  = 2,
         maxref  = 12,
         maxuref = 4,
         write   = True,
         npoints = 5,
         num     = 0.51,
         uref    = 0,): 

  datalog = treelog.DataLog('../results/square/images')

  methods = ['residual','goal','uniform']

  interest = zip([[np.pi,0]],['corner'])

  for poi, poitype in interest:

    for method in methods:

        domain, geom = mesh.rectilinear([numpy.linspace(0,np.pi,3),numpy.linspace(0,np.pi,3)])
        domain = domain.refine(uref)
        ns = function.Namespace()

        error_est  = []
        norm_L2    = []
        norm_H1    = []
        residual_e = []
        sum_ind    = []
        error_qoi  = []
        residual_z = []
        sum_goal   = []


        error_exact  = []
        error_qoi    = []
        error_est    = []
        sum_residual = []
        sum_goal     = []
        maxlvl       = []
        nelems       = []
        ndofs        = []
    
        for nref in range(maxref):
            
            log.info(method+' | '+poitype+' | Refinement :'+ str(nref))
    
            ns.basis = domain.basis('th-spline', degree=degree)
            ns.x     = geom
            x,y      = geom
            ns.pi    = np.pi

            #neumann BC
            ns.g = 'sin(x_1) cosh(x_0)'

            #exact solution
            ns.u = 'sin(x_1) sinh(x_0)'

            # Primal problem
            ns.uh = 'basis_n ?lhs_n'
            ns.e  = 'u - uh'
    
            A = domain.integrate(ns.eval_ij('basis_i,k basis_j,k d:x'), ischeme = 'gauss5')
            b = domain.boundary['right'].integrate(ns.eval_i('basis_i g d:x'), ischeme='gauss5')
    
            cons = domain.boundary['top,bottom,left'].project(0, onto=ns.basis, geometry=geom, ischeme='gauss5')
    
            lhs = A.solve(b, constrain = cons)
            ns = ns(lhs=lhs)
    
            # Dual problem
            dualspace = domain.refine(1)
            ns.dualbasis = dualspace.basis('th-spline', degree=degree)
    
            c  = 0.1 
            dx = poi[0] 
            dy = poi[1] 
    
            ns.q = function.exp(-((x-dx)**2+(y-dy)**2)/(2*c**2))
            B = dualspace.integrate(ns.eval_ij('dualbasis_i,k dualbasis_j,k d:x'), ischeme = 'gauss5')
            Q = dualspace.integrate(ns.eval_i('q dualbasis_i d:x'), ischeme='gauss5')
            
            cons = dualspace.boundary['top,bottom,left'].project(0, onto=ns.dualbasis, geometry=geom, ischeme='gauss5')
    
            ns.zvec = B.solve(Q, constrain = cons)
            ns.z    = 'dualbasis_n zvec_n'
            ns      = ns(lhs=lhs)
    
            ns.Iz   = dualspace.project(ns.z, ns.basis, geometry=geom, ischeme='gauss5') 
            ns.Iz   = 'basis_n Iz_n'

            # Collect indicators
            residual_indicators, res_int, res_jump, res_bound = elem_errors_residual(ns, geom, domain, dualspace, degree) 
            goal_indicators, goal_inter, goal_bound = elem_errors_goal(ns, geom, domain, dualspace, degree) 

            # Define error values for convergence plots
            try:
                maxlvl   += [len(domain.levels)]
            except:
                maxlvl   += [1]
            nelems       += [len(domain)]
            ndofs        += [len(ns.basis)]

            error_est    += [abs(domain.integrate('uh_,ii d:x' @ns, ischeme='gauss5'))]

            norm_L2      += [np.sqrt(domain.integrate('e^2 d:x' @ns, ischeme='gauss5'))]
            norm_H1      += [np.sqrt(domain.integrate('(e^2 + e_,i e_,i) d:x' @ns, ischeme='gauss5'))]
            residual_e   += [domain.integrate('e_,i e_,i d:x' @ns, ischeme='gauss5')]
            sum_ind      += [sum(residual_indicators.indicators.values())]

            error_qoi    += [abs(domain.integrate('(u q - uh q) d:x' @ns, ischeme='gauss5'))]
            residual_z   += [abs(dualspace.boundary['right'].integrate('(g ((z - Iz)^2)^.5) d:x' @ns, ischeme='gauss5') -
                             dualspace.integrate('(uh_,i ((z_,i - Iz_,i)^2)^.5) d:x' @ns, ischeme='gauss5'))]
            sum_goal     += [abs(sum(goal_indicators.indicators.values()))]
            
            print('Area of QoI: ',domain.integrate('q d:x' @ns, ischeme='gauss5'))

            # Refine mesh
            if method == 'residual':
                indicators = {'Indicators':residual_indicators.indicators,'Internal':res_int.indicators,'Interfaces':res_jump.indicators,'Boundary':res_bound.indicators}
                plotter.plot_indicators(method+'_indicators'+str(nref),domain, geom, indicators)
                domain = refiner.refine(domain, residual_indicators.abs(), num, maxlevel=8)

            if method == 'goal':
                indicators = {'Indicators':goal_indicators.indicators,'Internal':goal_inter.indicators,'Boundary':goal_bound.indicators}
                plotter.plot_indicators(method+'_indicators'+str(nref),domain, geom, indicators)
                domain = refiner.refine(domain, goal_indicators.abs(), num, maxlevel=8)

            if method == 'uniform':
                domain = domain.refine(1)
                if nref == maxuref:
                    break

            with treelog.add(datalog):
                plotter.plot_mesh('mesh_'+method+poitype+str(nref), domain, geom)

        plotter.plot_solution('solution_'+method+poitype+str(nref), domain, geom, ns.uh, alpha=1)
        plotter.plot_solution('exactsolution_'+method+poitype+str(nref), domain, geom, ns.u)

        if write:
            writer.write('../results/square/lshape'+method+poitype,
                        {'gausian width: c': c, 'degree': degree, 'uref': uref, 'maxuref': maxuref, 'nref': maxref, 'num': num, 'poi': poi},
                          maxlvl       = maxlvl,
                          norm_L2      = norm_L2,
                          norm_H1      = norm_H1,
                          residual_e   = residual_e,
                          residual_z   = residual_z,
                          sum_ind      = sum_ind,
                          error_qoi    = error_qoi,
                          error_est    = error_est,
                          sum_goal     = sum_goal,
                          nelems       = nelems,
                          ndofs        = ndofs,)

def elem_errors_residual(ns, geom, domain, dualspace, degree):

    # Residual-based error terms
    ns.rint    = 'uh_,ii'
    ns.rjump   = '.5 [[uh_,n]] n_n'
    ns.rbound  = 'g - uh_,n n_n'

    rint   = function.abs(ns.rint)
    rjump  = function.abs(ns.rjump)
    rbound = function.abs(ns.rbound)

    residual_indicators = indicater.elementbased(domain, geom, degree)
    int_indicators = indicater.elementbased(domain, geom, degree)
    jump_indicators = indicater.elementbased(domain, geom, degree)
    bound_indicators = indicater.elementbased(domain, geom, degree)

    residual_indicators = residual_indicators.residualbased(domain, rint, 'internal')
    int_indicators = int_indicators.residualbased(domain, rint, 'internal')

    residual_indicators = residual_indicators.residualbased(domain.interfaces, rjump, 'interface')
    jump_indicators = jump_indicators.residualbased(domain.interfaces, rjump, 'interface')

    residual_indicators = residual_indicators.residualbased(domain.boundary['right'], rbound, 'boundary')
    bound_indicators = bound_indicators.residualbased(domain.boundary['right'], rbound, 'boundary')

    return residual_indicators, int_indicators, jump_indicators, bound_indicators
 
def elem_errors_goal(ns, geom, domain, dualspace, degree):

    ns.gint   = '- uh_,i ((z_,i - Iz_,i)^2)^.5'
    ns.gbound = 'g ((z - Iz)^2)^.5'

    goal_indicators = indicater.elementbased(domain, geom, degree)
    int_indicators = indicater.elementbased(domain, geom, degree)
    jump_indicators = indicater.elementbased(domain, geom, degree)
    bound_indicators = indicater.elementbased(domain, geom, degree)

    goal_indicators = goal_indicators.goaloriented(dualspace, ns.gint, 'internal')
    int_indicators = int_indicators.goaloriented(dualspace, ns.gint, 'internal')

    goal_indicators = goal_indicators.goaloriented(dualspace.boundary['right'], ns.gbound, 'boundary')
    bound_indicators = bound_indicators.goaloriented(dualspace.boundary['right'], ns.gbound, 'boundary')

    return goal_indicators, int_indicators, bound_indicators
 
with config(verbose=3,nprocs=6):
    cli.run(main)
