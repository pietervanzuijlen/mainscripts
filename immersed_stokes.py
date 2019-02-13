from   nutils import *
import numpy as np
from utilities import *
import matplotlib.pyplot as plt
from matplotlib import collections

def main(degree     = 3,        # polynomial degree of pressure and velocity field
         uref       = 1,        # amount of uniform refinements before solving the problem
         refinements= 10,       # amount of refinement steps
         num        = 0.5,      # quantity indicating the amount of refinement each step
         nelem      = 4,        # amount of elements in the initial grid
         maxrefine  = 4,        # refinement level for the trimming operation and adaptivity
         maxuref    = 3,        # maximum amount of uniform refinements
         beta       = 14.,      # nitsche control parameter
         gamma      = .001,     # skeleton control parameter
         ):

    methods = ['goaloriented','residualbased','uniform']
    methods = ['goaloriented']

    nelems       = {} 
    ndofs        = {} 
    error_force  = {} 
    error_incomp = {} 

    # Defining grid 
    grid, geom = mesh.rectilinear([numpy.linspace(0,1,nelem+1),numpy.linspace(0,1,nelem+1)])
    grid = grid.refine(uref)

    # Defining trimmed domain
    rm = .2
    rc = .2
    M0 = .5
    M1 = .5

    x0, x1 = geom
    
    domain = grid.trim(function.norm2((x0,x1))-rc,         maxrefine=maxrefine)
    domain = domain.trim(function.norm2((x0-1,x1))-rc,     maxrefine=maxrefine)
    domain = domain.trim(function.norm2((x0-1,x1-1))-rc,   maxrefine=maxrefine)
    domain = domain.trim(function.norm2((x0,x1-1))-rc,     maxrefine=maxrefine)
    domain = domain.trim(function.norm2((x0-M0,x1-M1))-rm, maxrefine=maxrefine)

    for method in methods:

      nelems[method] = []
      ndofs[method] = []
      error_force[method] = []
      error_incomp[method] = []

      for nref in range(refinements):

        # Defining the background and skeleton 
        skeleton, ghost = domainmaker.skelghost(grid, domain)

        ns = function.Namespace()

        # Defining element sizes
        areas = domain.integrate_elementwise(function.J(geom), degree=degree)
        ifacelen = skeleton.integrate_elementwise(function.J(geom), degree=degree)

        ns.he = function.elemwise(domain.transforms, np.sqrt(areas))
        ns.hF = function.elemwise(skeleton.transforms, ifacelen)

        # Problem parameters
        ns.mu     = 1 
        ns.beta   = beta
        ns.gamma  = gamma / (degree * 2)
        ns.x      = geom
   
        ######################
        ### Primal problem ###
        ######################
       
        # Define bases
        ns.ubasis, ns.pbasis = function.chain([domain.basis('th-spline', degree=degree, continuity=degree-1).vector(2),
                                               domain.basis('th-spline', degree=degree, continuity=degree-1)])
    
        # Evaluation basis
        evalbasis = domain.basis('th-spline', degree=degree, continuity=degree-1)
        
        # Trail functions
        ns.u_i = 'ubasis_ni ?trail_n'
        ns.p = 'pbasis_n ?trail_n'
        
        # Test functions
        ns.v_i = 'ubasis_ni ?test_n'
        ns.q = 'pbasis_n ?test_n'
        
        # Stress
        ns.stress_ij = 'mu (u_i,j + u_j,i) - p δ_ij'

        # Inflow boundary condition
        ns.g_n = '-n_n'
    
        # Getting skeleton and ghost stabilization terms
        norm_derivative  = '({val}_,{i} n_{i})'
        jumpp = '(p_,a n_a)'
        jumpu = '(u_n,a n_a)'
        for i in 'bcdef'[:(degree-1)]:
            jumpp = norm_derivative.format(val=jumpp, i=i)
            jumpu = norm_derivative.format(val=jumpu, i=i)
            
        jumpq = '(q_,g n_g)'
        jumpv = '(v_n,g n_g)'
        for i in 'hijkl'[:(degree-1)]:
            jumpq = norm_derivative.format(val=jumpq, i=i)
            jumpv = norm_derivative.format(val=jumpv, i=i)
        
        ns.skeleton = '0.1 gamma mu^-1 hF^{} [[{}]] [[{}]]'.format(2*degree+1, jumpp, jumpq)
        ns.ghost    = 'gamma mu hF^{} [[{}]] [[{}]]'.format(2*degree-1, jumpu, jumpv)

        # nitsche term
        ns.nitsche  = 'mu ( (u_i,j + u_j,i) n_i) v_j + mu ( (v_i,j + v_j,i) n_i ) u_j - mu (beta / he) v_i u_i - p v_i n_i - q u_i n_i'

        # Defining the residual
        res = domain.boundary['left'].integral('(g_i v_i) d:x' @ ns, degree=degree*2)
        res += domain.integral('(- stress_ij v_i,j + q u_l,l) d:x' @ ns, degree=degree*2)
        res += domain.boundary['top,bottom,trimmed'].integral('nitsche d:x' @ns, degree=degree*2)
        res += skeleton.integral('skeleton d:x' @ns, degree=degree*2)
        res += ghost.integral('ghost d:x' @ns, degree=degree*2)
                
        # Solving the primal solution
        trail = solver.solve_linear('trail', res.derivative('test'))
        ns = ns(trail=trail) 
    
        ####################
        ### Dual problem ###
        ####################

        # Dual space by order elevation
        dualdegree = degree + 1
    
        # Define bases
        ns.zbasis, ns.sbasis = function.chain([domain.basis('th-spline', degree=dualdegree, continuity=dualdegree-1).vector(2),
                                               domain.basis('th-spline', degree=dualdegree, continuity=dualdegree-1)])
    
        # Trail functions
        ns.z_i = 'zbasis_ni ?dualtrail_n'
        ns.s = 'sbasis_n ?dualtrail_n'
        
        # Test functions
        ns.v_i = 'zbasis_ni ?dualtest_n'
        ns.q = 'sbasis_n ?dualtest_n'
        
    
        # Getting skeleton and ghost stabilization terms
        norm_derivative  = '({val}_,{i} n_{i})'
        jumps = '(s_,a n_a)'
        jumpz = '(z_n,a n_a)'
        for i in 'bcdef'[:(dualdegree-1)]:
            jumps = norm_derivative.format(val=jumps, i=i)
            jumpz = norm_derivative.format(val=jumpz, i=i)
    
        jumpq = '(q_,g n_g)'
        jumpv = '(v_n,g n_g)'
        for i in 'hijkl'[:(dualdegree-1)]:
            jumpq = norm_derivative.format(val=jumpq, i=i)
            jumpv = norm_derivative.format(val=jumpv, i=i)
        
        ns.gamma  = gamma / (dualdegree * 2)
        ns.skeleton = 'gamma mu^-1 hF^{} [[{}]] [[{}]]'.format(2*dualdegree+1, jumpq, jumps)
        ns.ghost    = 'gamma  mu hF^{} [[{}]] [[{}]]'.format(2*dualdegree-1, jumpv, jumpz)
    
        # nitsche term
        ns.nitsche  = 'mu ( (z_i,j + z_j,i) n_i) v_j + mu ( (v_i,j + v_j,i) n_i ) z_j - mu (beta / he) v_i z_i - s v_i n_i - q z_i n_i'

        # Defining residual for the dual problem
        res = domain.boundary['right'].integral('(n_i v_i) d:x' @ ns, degree=dualdegree*2)
        res += domain.integral('(-mu (z_j,i + z_i,j) v_j,i + s v_k,k + q z_l,l) d:x' @ ns, degree=dualdegree*2)
        res += domain.boundary['top,bottom,trimmed'].integral('nitsche d:x' @ns, degree=dualdegree*2)
        res += skeleton.integral('skeleton d:x' @ns, degree=dualdegree*2)
        res += ghost.integral('ghost d:x' @ns, degree=dualdegree*2)

        # Solving the dual problem
        dualtrail = solver.solve_linear('dualtrail', res.derivative('dualtest'))
        ns = ns(dualtrail=dualtrail) 

        # Calculating the projection values (no boundary condition included
        cons = domain.boundary['top,bottom,trimmed'].project(0, onto=ns.ubasis, geometry=geom, degree=degree*2)
        ns.Iz   = domain.projection(ns.z, ns.ubasis, geometry=geom, degree=dualdegree*2, constrain=cons)
        ns.Is   = domain.projection(ns.s, ns.pbasis, geometry=geom, degree=dualdegree*2)

        # Error values for convergence
        nelems[method]    += [len(domain)]
        ndofs[method]     += [len(ns.ubasis)]
        error_force[method]  += [domain.integrate(function.norm2('stress_ij,i d:x' @ns), ischeme='gauss5')]
        error_incomp[method] += [domain.integrate(function.abs('u_i,i d:x' @ns), ischeme='gauss5')]

        ######################
        ### Get indicators ###
        ######################

        # Define values
        ns.inflow  = '(g_i - stress_ij n_j) (g_i - stress_il n_l)'
        ns.outflow  = '(- stress_ij n_j) (- stress_il n_l)'
        ns.jump    = '[stress_ij] n_j [stress_il] n_l'
        ns.force   = 'stress_ij,j stress_il,l'
        ns.zsharp  = '(z_i - Iz_i) (z_i - Iz_i)'

        ns.incom   = '(u_i,i)^2'
        ns.ssharp  = '(s - Is)^2'

        h = np.sqrt(indicater.integrate(domain, geom, degree, 1, domain))

        # Get local indicators
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

        if method == 'goaloriented':

            # Assemble indicaters
            inter = incom*s_int + force*z_int
            iface = jump*z_jump
            bound = inflow*z_inflow + outflow*z_outflow
    
            indicators =  inter + iface + bound 
    
            # Plot indicators
            plotter.plot_indicators('indicators_'+method+'_'+str(nref), domain, geom, {'indicator':indicators,'unweighted_indicators':incom+force+jump+inflow,'weights':z_int+s_int+z_jump+z_inflow+z_outflow}, normalize=False, alpha=.5)
            #plotter.plot_indicators('residual_contributions_'+str(nref), domain, geom, {'force':force,'incompressibility':incom,'interfaces':jump,'boundaries':inflow+outflow}, normalize=False)
            #plotter.plot_indicators('sharp_contributions_'+str(nref), domain, geom, {'z_internal':z_int,'s_internal':s_int,'z_interfaces':z_jump,'z_boundaries':z_inflow+z_outflow}, normalize=False)
            #plotter.plot_indicators('indicators_'+'_'+str(nref), domain, geom, {'indicator':indicators,'internal':inter,'interfaces':iface,'boundaries':bound}, normalize=False)
    
            domain, grid, refined = refiner.refine(domain, indicators, num, evalbasis, maxlevel=maxrefine+uref, grid=grid, marker_type=None, select_type=None)

        if method == 'residualbased':

            # assemble indicaters
            inter = (incom + force)*h
            iface = jump*np.sqrt(h)
            bound = (inflow + outflow)*np.sqrt(h)

            indicators =  inter + iface + bound 

            plotter.plot_indicators('indicators_'+method+'_'+str(nref), domain, geom, {'indicator':indicators,'internal':inter,'interfaces':iface,'boundaries':bound}, normalize=False, alpha=.5)
            #plotter.plot_indicators('residual_contributions_'+str(nref), domain, geom, {'force':force*h,'incompressibility':incom*h,'interfaces':jump*np.sqrt(h),'boundaries':(inflow+outflow)*np.sqrt(h)}, normalize=False)
            #plotter.plot_indicators('indicators_'+method+'_'+str(nref), domain, geom, {'indicator':indicators})
            
            domain, grid, refined = refiner.refine(domain, indicators, num, evalbasis, maxlevel=maxrefine+uref, grid=grid, marker_type=None, select_type=None)

        if method == 'uniform':

            domain = domain.refine(1)
            grid = grid.refine(1)
            refined = True

            if nref == maxuref:
                refined = False                    

        # Stop with the refinement loop if nothing is refined
        if not refined:
            break

        #plotter.plot_levels(method+'mesh_'+str(nref), domain, geom, minlvl=uref)

        ns.momentum_i = 'mu (u_i,j + u_j,i)_,j + p_,i'
        #plotter.plot_streamlines('momentum',domain,geom,ns,ns.momentum)
        plotter.plot_streamlines('dualvelocity_'+str(nref), domain, geom, ns, ns.z)
        plotter.plot_solution('dualpressure'+str(nref), domain, geom, ns.s, alpha=0.3)
        plotter.plot_streamlines('velocity_'+str(nref), domain, geom, ns, ns.u)
        plotter.plot_solution('pressure'+str(nref), domain, geom, ns.p, alpha=0.3)
        plotter.plot_trim('mesh'+str(nref), domain, grid, geom)

        #plotter.plot_mesh('mesh'+str(nref),domain, geom)

    # Plot convergence
    plotter.plot_convergence('Estimated_error_force',ndofs,error_force,labels=['dofs','Estimated error'], slopemarker=True)
    plotter.plot_convergence('Estimated_error_incomp',ndofs,error_incomp,labels=['dofs','Estimated error'], slopemarker=True)

    #anouncer.drum()

with config(verbose=3,nprocs=6):
    cli.run(main)



