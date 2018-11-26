'''
In this code different forms of adaptivity are tested on an lshaped domain on which a laplace problem is solved. The exact solution to this problem is known. Goal-oriented, residual-based and uniform refinement will be compared.

'''

from   nutils import *
import numpy as np
from utilities import plotter
from utilities import indicater 
from utilities import writer
from utilities import domainmaker
from utilities import anouncer
from utilities import refiner

import treelog

def main(degree  = 2,
         poitype = 'center',
         poi     = [0,0],
         maxref  = 15,
         maxuref = 3,
         write   = True,
         npoints = 5,
         num     = 8,
         uref    = 1,): 

  datalog = treelog.DataLog('results/images/laplace')
  methods = ['residual','goal','uniform']
  methods = ['goal']

  with treelog.add(datalog):

    for method in methods:

        domain, geom = domainmaker.lshape(uref=uref)
        ns = function.Namespace()

        error_exact  = []
        error_qoi    = []
        error_est    = []
        sum_residual = []
        sum_goal     = []
        maxlvl       = []
        nelems       = []
    
        for nref in log.range('Refinement step: ', maxref):
    
            ns.basis = domain.basis('th-spline', degree=degree)
            ns.x     = geom

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

            # Primal problem
    
            ns.uh = 'basis_n ?lhs_n'
    
            A    = domain.integrate(ns.eval_ij('basis_i,k basis_j,k d:x'), ischeme = 'gauss5')
            b    = domain.boundary['patch1-top'].integrate(ns.eval_i('basis_i g1 d:x'), ischeme='gauss5')
            b   += domain.boundary['patch1-right'].integrate(ns.eval_i('basis_i g2 d:x'), ischeme='gauss5')
            b   += domain.boundary['patch0-right'].integrate(ns.eval_i('basis_i g3 d:x'), ischeme='gauss5')
            b   += domain.boundary['patch0-bottom'].integrate(ns.eval_i('basis_i g4 d:x'), ischeme='gauss5')
    
            cons = domain.boundary['patch0-left,patch1-left'].project(0, onto=ns.basis, geometry=geom, ischeme='gauss5')
    
            lhs = A.solve(b, constrain = cons)
            ns = ns(lhs=lhs)
    
            # Dual problem
    
            dualspace = domain.refine(1)
            ns.dualbasis = dualspace.basis('th-spline', degree=degree)
    
            amp  = 20 
            dx   = poi[0] 
            dy   = poi[1] 
    
            ns.q = (1+function.tanh(amp*(x-dx))) *(1+function.tanh(amp*(dx-x)))*(1+function.tanh(amp*(y-dy)))*(1+function.tanh(amp*(dy-y)))
            B = dualspace.integrate(ns.eval_ij('dualbasis_i,k dualbasis_j,k d:x'), ischeme = 'gauss5')
            Q = dualspace.integrate(ns.eval_i('q dualbasis_i d:x'), ischeme='gauss5')
            
            cons = dualspace.boundary['patch0-left,patch1-left'].project(0, onto=ns.dualbasis, geometry=geom, ischeme='gauss5')
    
            ns.zvec = B.solve(Q, constrain = cons)
            ns.z    = 'dualbasis_n zvec_n'
            ns      = ns(lhs=lhs)
    
            ns.Iz   = dualspace.project(ns.z, ns.basis, geometry=geom, ischeme='gauss4') 
            ns.Iz   = 'basis_n Iz_n'

            residual_indicators, res_int, res_jump, res_bound = elem_errors_residual(ns, geom, domain, dualspace, degree) 
            goal_indicators, goal_inter, goal_bound = elem_errors_goal(ns, geom, domain, dualspace, degree) 

            #residual_indicators = func_errors_residual(ns, geom, domain, dualspace, degree) 
            #goal_indicators = func_errors_goal(ns, geom, domain, dualspace, degree) 

            try:
                maxlvl   += [domain.levels]
            except:
                maxlvl   += [1]
            error_exact  += [domain.integrate(function.abs('(u - uh) d:x' @ns), ischeme='gauss5')]
            error_qoi    += [domain.integrate(function.abs('(u - uh) q d:x' @ns), ischeme='gauss5')]
            error_est    += [domain.integrate(function.abs('uh_,ii d:x' @ns), ischeme='gauss5')]
            sum_residual += [sum(residual_indicators.indicators.values())]
            sum_goal     += [sum(goal_indicators.indicators.values())]
            nelems       += [len(domain)]
   
            if method == 'residual':
                indicators = {'Indicators':residual_indicators.indicators,'Internal':res_int.indicators,'Interfaces':res_jump.indicators,'Boundary':res_bound.indicators}
                plotter.plot_indicators(method+'_indicators'+str(nref),domain, geom, indicators)
                domain = refiner.refine(domain, residual_indicators, num)

            if method == 'goal':
                indicators = {'Indicators':goal_indicators.indicators,'Internal':goal_inter.indicators,'Boundary':goal_bound.indicators}
                plotter.plot_indicators(method+'_indicators'+str(nref),domain, geom, indicators)
                plotter.plot_solution('dualsolution'+str(nref),domain, geom, ns.z)
                domain = refiner.refine(domain, goal_indicators, num)

            if method == 'uniform':
                domain = domain.refine(1)
                if nref == maxuref:
                    break

            plotter.plot_mesh('mesh'+str(nref), domain, geom)

        plotter.plot_mesh('mesh_'+method,domain,geom)

        if write:
            writer.write('results/lshape'+method+poitype,
                        {'degree': degree, 'nref': maxref, 'poi': poi},
                          error_exact  = error_exact,
                          error_qoi    = error_qoi,
                          error_est    = error_est,
                          sum_residual = sum_residual,
                          sum_goal     = sum_goal,
                          nelems       = nelems,)

def func_errors_residual(ns, geom, domain, dualspace, degree):

    indicators = indicater.functionbased(domain, geom, ns.basis, degree)
    
    ns.res = 'uh_,ii'

    indicators = indicators.add(domain, ns.basis, ns.res)

    return indicators

def func_errors_goal(ns, geom, domain, dualspace, degree):

    indicators = indicater.functionbased(dualspace, geom, ns.basis, degree)
    
    ns.resint  = '-uh_,i (z_,i - Iz_,i)'
    ns.resbou1 = 'g1 (z - Iz)'
    ns.resbou2 = 'g2 (z - Iz)'
    ns.resbou3 = 'g3 (z - Iz)'
    ns.resbou4 = 'g4 (z - Iz)'

    indicators = indicators.add(dualspace, ns.basis, ns.resint)
    indicators = indicators.add(dualspace.boundary['patch1-top'], ns.basis, ns.resbou1)
    indicators = indicators.add(dualspace.boundary['patch1-right'], ns.basis, ns.resbou2)
    indicators = indicators.add(dualspace.boundary['patch0-right'], ns.basis, ns.resbou3)
    indicators = indicators.add(dualspace.boundary['patch0-bottom'], ns.basis, ns.resbou4)

    return indicators

def elem_errors_residual(ns, geom, domain, dualspace, degree):

    # Residual-based error terms
    ns.rint    = 'uh_,ii'
    ns.rjump   = '.5 [[uh_,n]] n_n'
    ns.rbound1 = 'g1 - uh_,n n_n'
    ns.rbound2 = 'g2 - uh_,n n_n'
    ns.rbound3 = 'g3 - uh_,n n_n'
    ns.rbound4 = 'g4 - uh_,n n_n'

    rint    = function.abs(ns.rint)*function.J(geom)
    rjump   = function.abs(ns.rjump)*function.J(geom)  
    rbound1 = function.abs(ns.rbound1)*function.J(geom)
    rbound2 = function.abs(ns.rbound2)*function.J(geom)
    rbound3 = function.abs(ns.rbound3)*function.J(geom)
    rbound4 = function.abs(ns.rbound4)*function.J(geom)

    residual_indicators = indicater.elementbased(domain, geom, degree)
    int_indicators = indicater.elementbased(domain, geom, degree)
    jump_indicators = indicater.elementbased(domain, geom, degree)
    bound_indicators = indicater.elementbased(domain, geom, degree)

    residual_indicators = residual_indicators.residualbased(domain, rint, 'internal')
    int_indicators = int_indicators.residualbased(domain, rint, 'internal')

    residual_indicators = residual_indicators.residualbased(domain.interfaces, rjump, 'interface')
    jump_indicators = jump_indicators.residualbased(domain.interfaces, rjump, 'interface')

    residual_indicators = residual_indicators.residualbased(domain.boundary['patch1-top'], rbound1, 'boundary')
    residual_indicators = residual_indicators.residualbased(domain.boundary['patch1-right'], rbound2, 'boundary')
    residual_indicators = residual_indicators.residualbased(domain.boundary['patch0-right'], rbound3, 'boundary')
    residual_indicators = residual_indicators.residualbased(domain.boundary['patch0-bottom'], rbound4, 'boundary')
    bound_indicators = bound_indicators.residualbased(domain.boundary['patch1-top'], rbound1, 'boundary')
    bound_indicators = bound_indicators.residualbased(domain.boundary['patch1-right'], rbound2, 'boundary')
    bound_indicators = bound_indicators.residualbased(domain.boundary['patch0-right'], rbound3, 'boundary')
    bound_indicators = bound_indicators.residualbased(domain.boundary['patch0-bottom'], rbound4, 'boundary')

    return residual_indicators, int_indicators, jump_indicators, bound_indicators
 
def elem_errors_goal(ns, geom, domain, dualspace, degree):

    ns.gint    = '- uh_,i (z_,i - Iz_,i)'
    ns.gbound1 = '(g1) (z - Iz)'
    ns.gbound2 = '(g2) (z - Iz)'
    ns.gbound3 = '(g3) (z - Iz)'
    ns.gbound4 = '(g4) (z - Iz)'

    gint    = ns.gint*function.J(geom)
    gbound1 = ns.gbound1*function.J(geom)
    gbound2 = ns.gbound2*function.J(geom)
    gbound3 = ns.gbound3*function.J(geom)
    gbound4 = ns.gbound4*function.J(geom)

    goal_indicators = indicater.elementbased(domain, geom, degree)
    int_indicators = indicater.elementbased(domain, geom, degree)
    jump_indicators = indicater.elementbased(domain, geom, degree)
    bound_indicators = indicater.elementbased(domain, geom, degree)

    goal_indicators = goal_indicators.goaloriented(dualspace, gint, 'internal')
    int_indicators = int_indicators.goaloriented(dualspace, gint, 'internal')

    goal_indicators = goal_indicators.goaloriented(dualspace.boundary['patch1-top'], gbound1, 'boundary')
    goal_indicators = goal_indicators.goaloriented(dualspace.boundary['patch1-right'], gbound2, 'boundary')
    goal_indicators = goal_indicators.goaloriented(dualspace.boundary['patch0-right'], gbound3, 'boundary')
    goal_indicators = goal_indicators.goaloriented(dualspace.boundary['patch0-bottom'], gbound4, 'boundary')
    bound_indicators = bound_indicators.goaloriented(dualspace.boundary['patch1-top'], gbound1, 'boundary')
    bound_indicators = bound_indicators.goaloriented(dualspace.boundary['patch1-right'], gbound2, 'boundary')
    bound_indicators = bound_indicators.goaloriented(dualspace.boundary['patch0-right'], gbound3, 'boundary')
    bound_indicators = bound_indicators.goaloriented(dualspace.boundary['patch0-bottom'], gbound4, 'boundary')

    return goal_indicators, int_indicators, bound_indicators
 

cli.run(main)
