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
         maxref  = 10,
         maxuref = 3,
         write   = True,
         npoints = 5,
         num     = 0.5,
         uref    = 3,
         QoI     = 'singularity'): 

  datalog = treelog.DataLog('../results/laplace/images')
  methods = ['residual','goal','uniform']

  with treelog.add(datalog):

    for method in methods:

        domain, geom = domainmaker.lshape(uref=uref, width=2, height=2)
        ns = function.Namespace()

        # Values to save
        error_est   = []
        norm_L2     = []
        norm_H1     = []
        residual_e  = []
        sum_ind     = []
        error_qoi   = []
        residual_z  = []
        sum_goal    = []
        error_exact = []
        error_qoi   = []
        maxlvl      = []
        nelems      = []
        ndofs       = []
    
        for nref in range(maxref):
            
            log.info(method+' | Refinement :'+ str(nref))
    
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
            dualdegree = degree + 1
            ns.dualbasis = domain.basis('th-spline', degree=dualdegree, patchcontinuous=True, continuity=degree)

            ns.z = 'dualbasis_n ?duallhs_n'
    
            B = domain.integrate(ns.eval_ij('dualbasis_i,k dualbasis_j,k d:x'), degree=dualdegree*2)

            # Gausian data
            #c  = 0.05 
            #dx = 1 
            #dy = -1 
            #ns.q = function.exp(-((x-dx)**2+(y-dy)**2)/(2*c**2))
            #Q = domain.integrate(ns.eval_i('q dualbasis_i d:x'), degree=dualdegree*2)

            Q = domain.boundary['patch0-bottom'].boundary['right'].integrate(ns.eval_i('dualbasis_i d:x'), degree=dualdegree*2)

            consdual = domain.boundary['patch0-left,patch1-left'].project(0, onto=ns.dualbasis, geometry=geom, degree=dualdegree*2)
    
            duallhs = B.solve(Q, constrain = consdual)
            ns      = ns(duallhs=duallhs)
    
            ns.Iz   = domain.projection(ns.z, ns.basis, geometry=geom, degree=dualdegree*2, constrain=cons)

            # Collect indicators
            residual_indicators, res_int, res_jump, res_bound = elem_errors_residual(ns, geom, domain, degree) 
            goal_indicators, goal_inter, goal_jump, goal_bound, goal_sharp = elem_lnorm_goal(ns, geom, domain, degree)

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

            error_exact  += [abs(domain.integrate('(u - uh) d:x' @ns, degree=degree*2))]
            residual_z   += [domain.boundary['patch1-top'].integrate('g1 (z - Iz) d:x' @ns, degree=degree*2) + 
                             domain.boundary['patch1-right'].integrate('g2 (z - Iz) d:x' @ns, degree=degree*2) + 
                             domain.boundary['patch0-right'].integrate('g3 (z - Iz) d:x' @ns, degree=degree*2) + 
                             domain.boundary['patch0-bottom'].integrate('g4 (z - Iz) d:x' @ns, degree=degree*2) - 
                             domain.integrate('uh_,i (z - Iz)_,i d:x' @ns, degree=degree*2)]
            sum_goal     += [sum(goal_indicators.indicators.values())]
    
            Qref = domain.boundary['patch0-bottom'].boundary['right'].integrate('u d:x' @ ns , ischeme='gauss1')
            Q = domain.boundary['patch0-bottom'].boundary['right'].integrate('uh d:x' @ ns , ischeme='gauss1')
            error_qoi    += [Qref-Q]


            # Refine mesh
            if method == 'residual':
                indicators = {'Indicators absolute':residual_indicators.abs_indicators(),'Internal':res_int.indicators,'Interfaces':res_jump.indicators,'Boundary':res_bound.indicators}
                plotter.plot_indicators(method+'_indicators'+str(nref),domain, geom, indicators)
                domain = refiner.refine(domain, residual_indicators, num, maxlevel=6)

            if method == 'goal':
                indicators = {'Internal':goal_inter.indicators,'Boundary':goal_bound.indicators, 'Interface':goal_jump.indicators, '||z-Iz||':goal_sharp.indicators}
                plotter.plot_indicators(method+'_contributions'+str(nref),domain, geom, indicators)
                plotter.plot_indicators(method+'_indicators'+str(nref),domain, geom, {'Indicators absolute':goal_indicators.abs_indicators()})
                domain = refiner.refine(domain, goal_indicators, num, maxlevel=6)

            if method == 'uniform':
                domain = domain.refine(1)
                if nref == maxuref:
                    break

            plotter.plot_mesh('mesh_'+method+QoI+str(nref), domain, geom)


        if write:
            writer.write('../results/laplace/lshape'+method+QoI,
                        {'degree': degree, 'uref': uref, 'maxuref': maxuref, 'nref': maxref, 'num': num, 'QoI': QoI},
                          maxlvl       = maxlvl,
                          norm_L2      = norm_L2,
                          norm_H1      = norm_H1,
                          residual_e   = residual_e,
                          residual_z   = residual_z,
                          sum_ind      = sum_ind,
                          error_exact  = error_exact,
                          error_qoi    = error_qoi,
                          error_est    = error_est,
                          sum_goal     = sum_goal,
                          nelems       = nelems,
                          ndofs        = ndofs,)

    convergence(QoI, methods)

def convergence(QoI, methods):

    for error in ['norm_L2','norm_H1','residual_e','sum_ind','error_qoi','residual_z','sum_goal']:
        
        xval  = {} 
        yval  = {}
        level = {}

        for i, method in enumerate(methods):

            text = writer.read('../results/laplace/lshape'+methods[i]+QoI)
    
            xval[method]  = numpy.sqrt(text['ndofs'])
            yval[method]  = text[error]      

        labels = ['sqrt(ndofs)','error']

        plotter.plot_convergence(QoI+'-'+error, xval, yval, labels=labels, title=error)
        
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

    residual_indicators  = indicater.elementbased(domain, geom, degree, dualspacetype='k-refined')
    int_indicators   = indicater.elementbased(domain, geom, degree, dualspacetype='k-refined')
    jump_indicators  = indicater.elementbased(domain, geom, degree, dualspacetype='k-refined')
    bound_indicators = indicater.elementbased(domain, geom, degree, dualspacetype='k-refined')

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


def elem_errors_goal(ns, geom, domain, degree):

    ns.gint    = '-uh_,i abs(z_,i - Iz_,i)'
    ns.gbound1 = 'g1 abs(z - Iz)'
    ns.gbound2 = 'g2 abs(z - Iz)'
    ns.gbound3 = 'g3 abs(z - Iz)'
    ns.gbound4 = 'g4 abs(z - Iz)'

    #ns.gint    = 'uh_,ii (z - Iz)'
    #ns.gbound1 = '(g1 - uh_,i n_i) (z - Iz)'
    #ns.gbound2 = '(g2 - uh_,i n_i) (z - Iz)'
    #ns.gbound3 = '(g3 - uh_,i n_i) (z - Iz)'
    #ns.gbound4 = '(g4 - uh_,i n_i) (z - Iz)'

    goal_indicators  = indicater.elementbased(domain, geom, degree, dualspacetype='k-refined')
    int_indicators   = indicater.elementbased(domain, geom, degree, dualspacetype='k-refined')
    bound_indicators = indicater.elementbased(domain, geom, degree, dualspacetype='k-refined')

    goal_indicators.goaloriented(domain, ns.gint, 'internal')
    int_indicators.goaloriented(domain, ns.gint, 'internal')

    goal_indicators.goaloriented(domain.boundary['patch1-top'], ns.gbound1, 'boundary')
    goal_indicators.goaloriented(domain.boundary['patch1-right'], ns.gbound2, 'boundary')
    goal_indicators.goaloriented(domain.boundary['patch0-right'], ns.gbound3, 'boundary')
    goal_indicators.goaloriented(domain.boundary['patch0-bottom'], ns.gbound4, 'boundary')

    bound_indicators.goaloriented(domain.boundary['patch1-top'], ns.gbound1, 'boundary')
    bound_indicators.goaloriented(domain.boundary['patch1-right'], ns.gbound2, 'boundary')
    bound_indicators.goaloriented(domain.boundary['patch0-right'], ns.gbound3, 'boundary')
    bound_indicators.goaloriented(domain.boundary['patch0-bottom'], ns.gbound4, 'boundary')

    return goal_indicators, int_indicators, bound_indicators 
 
def elem_lnorm_goal(ns, geom, domain, degree):

    # Residual-based error terms
    ns.rint    = 'abs(uh_,ii)'
    ns.rjump   = 'abs(.5 [[uh_,n]] n_n)'
    ns.rbound1 = 'abs(g1 - uh_,n n_n)'
    ns.rbound2 = 'abs(g2 - uh_,n n_n)'
    ns.rbound3 = 'abs(g3 - uh_,n n_n)'
    ns.rbound4 = 'abs(g4 - uh_,n n_n)'
    ns.rz      = 'abs(z - Iz)'

    goal_indicators = indicater.elementbased(domain, geom, degree, dualspacetype='k-refined')
    int_indicators      = indicater.elementbased(domain, geom, degree, dualspacetype='k-refined')
    jump_indicators     = indicater.elementbased(domain, geom, degree, dualspacetype='k-refined')
    bound_indicators    = indicater.elementbased(domain, geom, degree, dualspacetype='k-refined')
    sharp_indicators    = indicater.elementbased(domain, geom, degree, dualspacetype='k-refined')

    sharp_indicators.goaloriented(domain, ns.rz, 'internal')

    goal_indicators.goaloriented(domain, ns.rint*ns.rz, 'internal')
    int_indicators.goaloriented(domain, ns.rint*ns.rz, 'internal')

    goal_indicators.goaloriented(domain.interfaces, ns.rjump*ns.rz, 'interface')
    jump_indicators.goaloriented(domain.interfaces, ns.rjump*ns.rz, 'interface')

    goal_indicators.goaloriented(domain.boundary['patch1-top'], ns.rbound1*ns.rz, 'boundary')
    goal_indicators.goaloriented(domain.boundary['patch1-right'], ns.rbound2*ns.rz, 'boundary')
    goal_indicators.goaloriented(domain.boundary['patch0-right'], ns.rbound3*ns.rz, 'boundary')
    goal_indicators.goaloriented(domain.boundary['patch0-bottom'], ns.rbound4*ns.rz, 'boundary')

    bound_indicators.goaloriented(domain.boundary['patch1-top'], ns.rbound1*ns.rz, 'boundary')
    bound_indicators.goaloriented(domain.boundary['patch1-right'], ns.rbound2*ns.rz, 'boundary')
    bound_indicators.goaloriented(domain.boundary['patch0-right'], ns.rbound3*ns.rz, 'boundary')
    bound_indicators.goaloriented(domain.boundary['patch0-bottom'], ns.rbound4*ns.rz, 'boundary')

    return goal_indicators, int_indicators, jump_indicators, bound_indicators, sharp_indicators
 

with config(verbose=3,nprocs=6):
    cli.run(main)
