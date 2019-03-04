from nutils import * 
import numpy as np
from utilities import *

def main(nelem      = 10,
         maxrefine  = 3,
         uref       = 2,
         degree     = 2,):

    ns = function.Namespace()
    
    ### Create mesh ###
    domain, geom = mesh.rectilinear([numpy.linspace(-2,2,nelem+1),numpy.linspace(-2,2,nelem+1)])
    domain, geom = domainmaker.rotategrid(domain,geom,np.pi*0.1)

    domain = domain.refine(uref)

    x0, x1 = geom
    th = np.pi/4 - function.ArcTan2(x0-x1,x0+x1) 

    domain = domain.trim(x0+1, maxrefine=maxrefine, name='left')
    domain = domain.trim(-x1+1, maxrefine=maxrefine, name='top')
    domain = domain.trim(-x0+1, maxrefine=maxrefine, name='right')
    domain = domain.trim(x1+1, maxrefine=maxrefine, name='bottom')

    domain = domain.trim(function.max(x0,x1), maxrefine=maxrefine, name='inner')

    ### Exact solution ###

    R  = (x0**2 + x1**2)**.5
    ns.u  = R**(2/3) * function.Sin(2/3*th+np.pi/3) 

    ### Primal problem ###
    ns.x = geom

    ns.basis = domain.basis('th-spline', degree=degree)

    #neumann BC
    ns.g1 = -2 * (x0*function.sin((2*th+np.pi)/3) - x1*function.cos((2*th+np.pi)/3))/(3*(x0**2+x1**2)**(2/3))
    ns.g2 =  2 * (x1*function.sin((2*th+np.pi)/3) + x0*function.cos((2*th+np.pi)/3))/(3*(x0**2+x1**2)**(2/3))
    ns.g3 =  2 * (x0*function.sin((2*th+np.pi)/3) - x1*function.cos((2*th+np.pi)/3))/(3*(x0**2+x1**2)**(2/3))
    ns.g4 = -2 * (x1*function.sin((2*th+np.pi)/3) + x0*function.cos((2*th+np.pi)/3))/(3*(x0**2+x1**2)**(2/3))

    ns.uh = 'basis_n ?lhs_n'
    ns.v = 'basis_n ?test_n'

    # Nitsche
    ns.beta = 10
    areas = domain.integrate_elementwise(function.J(geom), degree=degree)
    ns.he = function.elemwise(domain.transforms, np.sqrt(areas))
    ns.nitsche  = ' (uh_,i  n_i) v + ( v_,i n_i ) u - (beta / he) v uh'
    
    res = domain.integral('uh_,i v_,i d:x' @ ns, degree=degree*2) 
    res -= domain.boundary['left'].integral('g1 v d:x' @ ns, degree=degree*2) 
    res -= domain.boundary['top'].integral('g2 v d:x' @ ns, degree=degree*2) 
    res -= domain.boundary['right'].integral('g3 v d:x' @ ns, degree=degree*2) 
    res -= domain.boundary['bottom'].integral('g4 v d:x' @ ns, degree=degree*2) 
    res -= domain.boundary['inner'].integral('nitsche d:x' @ ns, degree=degree*2)

    lhs = solver.solve_linear('lhs', res.derivative('test'))
    ns = ns(lhs=lhs)

    plotter.plot_solution('solution',domain,geom,ns.uh,alpha=0.5)
    plotter.plot_solution('exactsolution',domain,geom,ns.u)

cli.run(main)
