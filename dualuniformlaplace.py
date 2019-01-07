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
from utilities import anouncer

import treelog

def main(degree  = 1,
         maxref  = 4,
         maxuref = 2,
         write   = True,
         npoints = 5,
         num     = 0.2,
         uref    = 3,): 

  datalog = treelog.DataLog('../results/laplace/images')

  #methods = ['residual','goal','uniform']
  methods = ['goal']

  qois = [[.5,-.5]]
  qoitypes = ['corner']

  for qoi, qoitype in zip(qois, qoitypes):

    for method in methods:

        domain, geom = domainmaker.lshape(uref=uref)
        ns = function.Namespace()

        error_est  = []
        norm_L2    = []
        norm_H1    = []
        residual_e = []
        sum_ind    = []
        error_qoi  = []
        residual_z = []
        sum_goal   = []

        qoi_area     = []
        error_exact  = []
        error_qoi    = []
        error_est    = []
        sum_residual = []
        sum_goal     = []
        maxlvl       = []
        nelems       = []
        ndofs        = []
    
        for nref in range(maxref):
            
            log.info(method+' | '+qoitype+' | Refinement :'+ str(nref))
    
            ns.basis = domain.basis('th-spline', degree=degree, patchcontinuous=True, continuity=degree-1)
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
            ns.e  = 'u - uh'
    
            A    = domain.integrate(ns.eval_ij('basis_i,k basis_j,k d:x'), degree=degree*2)
            b    = domain.boundary['patch1-top'].integrate(ns.eval_i('basis_i g1 d:x'), degree=degree*2)
            b   += domain.boundary['patch1-right'].integrate(ns.eval_i('basis_i g2 d:x'), degree=degree*2)
            b   += domain.boundary['patch0-right'].integrate(ns.eval_i('basis_i g3 d:x'), degree=degree*2)
            b   += domain.boundary['patch0-bottom'].integrate(ns.eval_i('basis_i g4 d:x'), degree=degree*2)
    
            cons = domain.boundary['patch0-left,patch1-left'].project(0, onto=ns.basis, geometry=geom, degree=degree*2)
    
            lhs = A.solve(b, constrain = cons)
            ns = ns(lhs=lhs)
    
            # Dual problem

            dualdomain = domain.refine(1)
            ns.dualbasis = dualdomain.basis('th-spline', degree=degree, patchcontinuous=True)

            ns.z = 'dualbasis_n ?duallhs_n'
    
            c  = 0.05 
            dx = qoi[0] 
            dy = qoi[1] 
    
            ns.q = function.exp(-((x-dx)**2+(y-dy)**2)/(2*c**2))
            B = dualdomain.integrate(ns.eval_ij('dualbasis_i,k dualbasis_j,k d:x'), degree=degree*2)
            #Q = dualdomain.integrate(ns.eval_i('q dualbasis_i d:x'), degree=degree*2)
            
            Q = dualdomain.boundary['patch0-bottom'].boundary['right'].integrate(ns.eval_i('dualbasis_i d:x'), degree=degree*2)

            consdual = dualdomain.boundary['patch0-left,patch1-left'].project(0, onto=ns.dualbasis, geometry=geom, degree=degree*2)
    
            duallhs = B.solve(Q, constrain = consdual)
            #ns.z    = 'dualbasis_n zvec_n'
            ns      = ns(duallhs=duallhs)
    
            #ns.Iz   = dualdomain.projection(ns.z, ns.basis, geometry=geom, degree=degree*2, constrain=cons, ptype='convolute') 
            ns.Iz   = dualdomain.projection(ns.z, ns.basis, geometry=geom, degree=degree*2, constrain=cons) 

            # Collect indicators
            residual_indicators, res_int, res_jump, res_bound = elem_errors_residual(ns, geom, dualdomain, degree) 
            goal_indicators, goal_inter, goal_bound, sharpsolution = elem_errors_goal(ns, geom, domain, dualdomain, degree)

            #residual_indicators = func_errors_residual(ns, geom, dualdomain, degree)
            #goal_indicators = func_errors_goal(ns, geom, dualdomain, degree)

            # Define error values for convergence plots
            try:
                maxlvl   += [len(domain.levels)]
            except:
                maxlvl   += [1]

            nelems       += [len(domain)]
            ndofs        += [len(ns.basis)]

            error_est    += [abs(domain.integrate('uh_,ii d:x' @ns, degree=degree*2))]

            norm_L2      += [np.sqrt(domain.integrate('e^2 d:x' @ns, degree=degree*2))]
            norm_H1      += [np.sqrt(domain.integrate('(e^2 + e_,i e_,i) d:x' @ns, degree=degree*2))]
            residual_e   += [domain.integrate('e_,i e_,i d:x' @ns, degree=degree*2)]
            sum_ind      += [sum(residual_indicators.indicators.values())]

            qoi_area     += [abs(domain.integrate('q d:x' @ns, degree=degree*2))]
            error_exact  += [abs(domain.integrate('(u - uh) d:x' @ns, degree=degree*2))]
            residual_z   += [dualdomain.boundary['patch1-top'].integrate('g1 (z - Iz) d:x' @ns, degree=degree*2) + 
                             dualdomain.boundary['patch1-right'].integrate('g2 (z - Iz) d:x' @ns, degree=degree*2) + 
                             dualdomain.boundary['patch0-right'].integrate('g3 (z - Iz) d:x' @ns, degree=degree*2) + 
                             dualdomain.boundary['patch0-bottom'].integrate('g4 (z - Iz) d:x' @ns, degree=degree*2) - 
                             dualdomain.integrate('uh_,i (z - Iz)_,i d:x' @ns, degree=degree*2)]
            sum_goal     += [sum(goal_indicators.indicators.values())]


            #Qref =  domain.refine(3).integrate('u q d:x' @ns, degree=degree*2)
            Qref = domain.boundary['patch0-bottom'].boundary['right'].integrate('u d:x' @ ns , ischeme='gauss1')
            log.user('Qref :', Qref)

            #Q =  domain.integrate('uh q d:x' @ns, degree=degree*2)
            Q = domain.boundary['patch0-bottom'].boundary['right'].integrate('uh d:x' @ ns , ischeme='gauss1')
            log.user('Qref - Q:', Qref-Q)

            error_qoi    += [Qref-Q]

            log.user('R(z-Iz) :', residual_z[-1])

            log.user('sum :', sum(goal_indicators.indicators.values()))

            plotter.plot_solution('dualsolution'+str(nref), dualdomain, geom, ns.z)
            plotter.plot_solution('projected_dualsolution'+str(nref), domain, geom, ns.Iz)
            plotter.plot_solution('projection'+str(nref), dualdomain, geom, ns.z-ns.Iz, grid=domain, cmap='gist_heat', alpha=0.5)

            ns.gradz = '(z - Iz)_,i n_i'
            plotter.plot_solution('grad_projection'+str(nref), dualdomain, geom, ns.gradz, grid=domain, cmap='gist_heat', alpha=0.5)

            sharpind = {'Sharpsolution':sharpsolution.indicators}
            plotter.plot_indicators('Sharp'+str(nref),domain, geom, sharpind)

            # Refine mesh
            if method == 'residual':
                indicators = {'Indicators absolute':residual_indicators.indicators,'Internal':res_int.indicators,'Interfaces':res_jump.indicators,'Boundary':res_bound.indicators}
                plotter.plot_indicators(method+'_indicators'+str(nref),domain, geom, indicators)
                domain = refiner.refine(domain, residual_indicators, num, maxlevel=8)

            if method == 'goal':
                indicators = {'Indicators absolute':goal_indicators.abs_indicators(),'Internal':goal_inter.indicators,'Boundary':goal_bound.indicators}
                plotter.plot_indicators(method+'_indicators'+str(nref),domain, geom, indicators)
                domain = refiner.refine(domain, goal_indicators, num, maxlevel=8)

            if method == 'uniform':
                domain = domain.refine(1)
                if nref == maxuref:
                    break

            with treelog.add(datalog):
                plotter.plot_mesh('mesh_'+method+qoitype+str(nref), domain, geom)

        if write:
            writer.write('../results/laplace/lshape'+method+qoitype,
                        {'gausian width: c': c, 'degree': degree, 'uref': uref, 'maxuref': maxuref, 'nref': maxref, 'num': num, 'qoi': qoi},
                          maxlvl       = maxlvl,
                          norm_L2      = norm_L2,
                          norm_H1      = norm_H1,
                          residual_e   = residual_e,
                          residual_z   = residual_z,
                          sum_ind      = sum_ind,
                          qoi_area     = qoi_area,
                          error_exact  = error_exact,
                          error_qoi    = error_qoi,
                          error_est    = error_est,
                          sum_goal     = sum_goal,
                          nelems       = nelems,
                          ndofs        = ndofs,)

  convergence(qoitypes, methods)

def convergence(qoitypes, methods):

    for qoitype in qoitypes: 
        for error in ['norm_L2','norm_H1','residual_e','sum_ind','error_qoi','residual_z','sum_goal']:
            
            xval  = {} 
            yval  = {}
            level = {}

            for i, method in enumerate(methods):

                text = writer.read('../results/laplace/lshape'+methods[i]+qoitype)
        
                xval[method]  = numpy.sqrt(text['ndofs'])
                yval[method]  = text[error]      

            labels = ['sqrt(ndofs)','error']

            plotter.plot_convergence(qoitype+'-'+error, xval, yval, labels=labels, title=error)
        
def func_errors_residual(ns, geom, domain, degree):

    indicators = indicater.functionbased(domain, geom, ns.basis, degree)
    
    ns.res = 'uh_,ii'

    indicators = indicators.add(domain, ns.basis, ns.res)

    return indicators

def func_errors_goal(ns, geom, domain, degree):

    indicators = indicater.functionbased(domain, geom, ns.basis, degree)
    
    ns.resint  = '-uh_,i (z_,i - Iz_,i)'
    ns.resbou1 = 'g1 (z - Iz)'
    ns.resbou2 = 'g2 (z - Iz)'
    ns.resbou3 = 'g3 (z - Iz)'
    ns.resbou4 = 'g4 (z - Iz)'

    indicators = indicators.add(domain, ns.basis, ns.resint)
    indicators = indicators.add(domain.boundary['patch1-top'], ns.basis, ns.resbou1)
    indicators = indicators.add(domain.boundary['patch1-right'], ns.basis, ns.resbou2)
    indicators = indicators.add(domain.boundary['patch0-right'], ns.basis, ns.resbou3)
    indicators = indicators.add(domain.boundary['patch0-bottom'], ns.basis, ns.resbou4)

    return indicators

def elem_errors_residual(ns, geom, domain, degree):

    # Residual-based error terms
    ns.rint    = 'uh_,ii'
    ns.rjump   = '.5 [[uh_,n]] n_n'
    ns.rbound1 = 'g1 - uh_,n n_n'
    ns.rbound2 = 'g2 - uh_,n n_n'
    ns.rbound3 = 'g3 - uh_,n n_n'
    ns.rbound4 = 'g4 - uh_,n n_n'

    rint    = function.abs(ns.rint)
    rjump   = function.abs(ns.rjump)
    rbound1 = function.abs(ns.rbound1)
    rbound2 = function.abs(ns.rbound2)
    rbound3 = function.abs(ns.rbound3)
    rbound4 = function.abs(ns.rbound4)

    residual_indicators = indicater.elementbased(domain, geom, degree, dualspacetype='p-refined')
    int_indicators      = indicater.elementbased(domain, geom, degree, dualspacetype='p-refined')
    jump_indicators     = indicater.elementbased(domain, geom, degree, dualspacetype='p-refined')
    bound_indicators    = indicater.elementbased(domain, geom, degree, dualspacetype='p-refined')

    residual_indicators.residualbased(domain, rint, 'internal')
    int_indicators.residualbased(domain, rint, 'internal')

    residual_indicators.residualbased(domain.interfaces, rjump, 'interface')
    jump_indicators.residualbased(domain.interfaces, rjump, 'interface')

    residual_indicators.residualbased(domain.boundary['patch1-top'], rbound1, 'boundary')
    residual_indicators.residualbased(domain.boundary['patch1-right'], rbound2, 'boundary')
    residual_indicators.residualbased(domain.boundary['patch0-right'], rbound3, 'boundary')
    residual_indicators.residualbased(domain.boundary['patch0-bottom'], rbound4, 'boundary')

    bound_indicators.residualbased(domain.boundary['patch1-top'], rbound1, 'boundary')
    bound_indicators.residualbased(domain.boundary['patch1-right'], rbound2, 'boundary')
    bound_indicators.residualbased(domain.boundary['patch0-right'], rbound3, 'boundary')
    bound_indicators.residualbased(domain.boundary['patch0-bottom'], rbound4, 'boundary')

    return residual_indicators, int_indicators, jump_indicators, bound_indicators
 
def elem_errors_goal(ns, geom, domain, dualdomain, degree):

    ns.gint    = '-uh_,i (z - Iz)_,i'
    ns.gbound1 = 'g1 (z - Iz)'
    ns.gbound2 = 'g2 (z - Iz)'
    ns.gbound3 = 'g3 (z - Iz)'
    ns.gbound4 = 'g4 (z - Iz)'
    ns.sharp   = '(z - Iz)'

    goal_indicators  = indicater.elementbased(domain, geom, degree, dualspacetype='p-refined')
    int_indicators   = indicater.elementbased(domain, geom, degree, dualspacetype='p-refined')
    bound_indicators = indicater.elementbased(domain, geom, degree, dualspacetype='p-refined')
    sharpsolution    = indicater.elementbased(domain, geom, degree, dualspacetype='p-refined')

    goal_indicators.goaloriented(dualdomain, ns.gint, 'internal')
    int_indicators.goaloriented(dualdomain, ns.gint, 'internal')
    sharpsolution.goaloriented(dualdomain, ns.sharp, 'internal')

    goal_indicators.goaloriented(dualdomain.boundary['patch1-top'], ns.gbound1, 'boundary')
    goal_indicators.goaloriented(dualdomain.boundary['patch1-right'], ns.gbound2, 'boundary')
    goal_indicators.goaloriented(dualdomain.boundary['patch0-right'], ns.gbound3, 'boundary')
    goal_indicators.goaloriented(dualdomain.boundary['patch0-bottom'], ns.gbound4, 'boundary')

    bound_indicators.goaloriented(dualdomain.boundary['patch1-top'], ns.gbound1, 'boundary')
    bound_indicators.goaloriented(dualdomain.boundary['patch1-right'], ns.gbound2, 'boundary')
    bound_indicators.goaloriented(dualdomain.boundary['patch0-right'], ns.gbound3, 'boundary')
    bound_indicators.goaloriented(dualdomain.boundary['patch0-bottom'], ns.gbound4, 'boundary')

    return goal_indicators, int_indicators, bound_indicators, sharpsolution
 
with config(verbose=3,nprocs=6):
    cli.run(main)
