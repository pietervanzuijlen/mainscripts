from nutils import *
from utilities import *
import numpy as np

def main(degree      = 3,
         uref        = 0,
         refinements = 10,
         num         = 0.5,
         maxrefine   = 4,
         maxuref     = 3,
         beta        = 500,):

    positions = [.5,.4]
    rc = 0.2

    methods = ['goaloriented','residualbased','uniform']
    #methods = ['uniform']
    

    for M1 in positions:

      nelems        = {} 
      ndofs         = {} 
      error_force   = {} 
      error_incomp  = {} 
      error_qoi     = {} 

      for method in methods:

        nelems[method]       = []
        ndofs[method]        = []
        error_force[method]  = []
        error_incomp[method] = []
        error_qoi[method]    = []
        
        domain, geom = domainmaker.porous(uref=uref, M1=M1)

        ns = function.Namespace()
        ns.beta = beta 
        ns.mu = 1
        ns.x  = geom

        for nref in range(refinements):
            
          ######################
          ### Primal problem ###
          ######################

          # Define bases
          ns.ubasis, ns.pbasis = function.chain([domain.basis('th-spline', degree=degree, patchcontinuous=True, continuity=degree-2).vector(2),
                                                               domain.basis('th-spline', degree=degree-1, patchcontinuous=True, continuity=degree-2)])
    
          # Evaluation basis 
          evalbasis = domain.basis('th-spline', degree=degree)
    
          # Trail functions
          ns.u_i = 'ubasis_ni ?trail_n'
          ns.p = 'pbasis_n ?trail_n'
          
          # Test functions
          ns.v_i = 'ubasis_ni ?test_n'
          ns.q = 'pbasis_n ?test_n'
    
          # Stress
          ns.stress_ij = 'mu (u_i,j + u_j,i) - p δ_ij'
    
          # Poiseulle inflow
          ns.rc = rc
          ns.uin_i = '<4 (x_1 - rc) ( 1 - rc - x_1 ), 0>_i'
    
          # Define one pressure value 
          cons = util.NanVec(len(ns.pbasis))
          cons[-1] = 0
    
          # Nitsche values
          areas = domain.integrate_elementwise(function.J(geom), degree=degree)
          ns.he = function.elemwise(domain.transforms, np.sqrt(areas))
          ns.noslip  = 'mu ( (u_i,j + u_j,i) n_i) v_j + mu ( (v_i,j + v_j,i) n_i ) u_j - mu (beta / he) v_i u_i - p v_i n_i - q u_i n_i'
          ns.inflow  = 'mu ( (u_i,j + u_j,i) n_i) v_j + mu ( (v_i,j + v_j,i) n_i ) (u_j - uin_j) - mu (beta / he) v_i (u_i - uin_i) - p v_i n_i - q (u_i - uin_i) n_i'
    
          res = domain.integral('(-stress_ij v_i,j + q u_l,l) d:x' @ ns, degree=degree*2)
          res += domain.boundary['top,bottom,corners,circle'].integral('noslip d:x' @ns, degree=degree*2)
          res += domain.boundary['left'].integral('inflow d:x' @ns, degree=degree*2)
    
          trail = solver.solve_linear('trail', res.derivative('test'))
          ns = ns(trail=trail) 

          ######################
          ### Dual problem ###
          ######################

          dualdegree = degree + 1

          ns.zbasis, ns.sbasis = function.chain([domain.basis('th-spline', degree=dualdegree, patchcontinuous=True, continuity=dualdegree-2).vector(2),
                                                               domain.basis('th-spline', degree=dualdegree-1, patchcontinuous=True, continuity=dualdegree-2)])
    
          # Trail functions
          ns.z_i = 'zbasis_ni ?dualtrail_n'
          ns.s = 'sbasis_n ?dualtrail_n'
          
          # Test functions
          ns.v_i = 'zbasis_ni ?dualtest_n'
          ns.q = 'sbasis_n ?dualtest_n'
    
          # Stress
          ns.dualstress_ij = 'mu (z_i,j + z_j,i) - s δ_ij'
    
          # Define one pressure value 
          cons = util.NanVec(len(ns.sbasis))
          cons[-1] = 0
    
          # Nitsche values
          ns.beta = beta 
          areas = domain.integrate_elementwise(function.J(geom), degree=dualdegree)
          ns.he = function.elemwise(domain.transforms, np.sqrt(areas))
          ns.noslip  = 'mu ( (z_i,j + z_j,i) n_i) v_j + mu ( (v_i,j + v_j,i) n_i ) z_j - mu (beta / he) v_i z_i - s v_i n_i - q z_i n_i'
          ns.inflow  = 'mu ( (z_i,j + z_j,i) n_i) v_j + mu ( (v_i,j + v_j,i) n_i ) (z_j - uin_j) - mu (beta / he) v_i (z_i - uin_i) - s v_i n_i - q (z_i - uin_i) n_i'
    
          res = domain.integral('(-dualstress_ij v_i,j + q z_l,l) d:x' @ ns, degree=degree*2)
          res += domain.boundary['top,bottom,corners,circle'].integral('noslip d:x' @ns, degree=degree*2)
          res += domain.boundary['left'].integral('v_i n_i d:x' @ns, degree=degree*2)
    
          dualtrail = solver.solve_linear('dualtrail', res.derivative('dualtest'))
          ns = ns(dualtrail=dualtrail) 

          #cons = domain.boundary['top,bottom,corners,circle'].project(0, onto=ns.ubasis, geometry=geom, degree=degree*2)
          ns.Iz   = domain.projection(ns.z, ns.ubasis, geometry=geom, degree=degree*2, droptol=0)
          ns.Is   = domain.projection(ns.s, ns.pbasis, geometry=geom, degree=degree*2, droptol=0)
          
          ### Get errors ###
          nelems[method]       += [len(domain)]
          ndofs[method]        += [len(ns.ubasis)]
          error_force[method]  += [domain.integrate(function.norm2('stress_ij,i d:x' @ns), ischeme='gauss5')]
          error_incomp[method] += [domain.integrate(function.abs('u_i,i d:x' @ns), ischeme='gauss5')]
          error_qoi[method]    += [abs(domain.boundary['right'].integrate('n_i u_i d:x' @ ns, ischeme='gauss5'))]
          ### Get errors ###
    
    
          ### Get indicators ###
          ns.jump    = '[stress_ij] n_j [stress_il] n_l'
          ns.force   = 'stress_ij,j stress_il,l'
          ns.outflow = '(stress_ij n_j) (stress_il n_l) '
          ns.zsharp  = '(z_i - Iz_i) (z_i - Iz_i)'

          ns.incom   = '(u_i,i)^2'
          ns.ssharp  = '(s - Is)^2'

          h = np.sqrt(indicater.integrate(domain, geom, degree, 1, domain))

          incom   = np.sqrt(indicater.integrate(domain, geom, degree, ns.incom, domain))
          force   = np.sqrt(indicater.integrate(domain, geom, degree, ns.force, domain))
          outflow = np.sqrt(indicater.integrate(domain, geom, degree, ns.outflow, domain.boundary['right']))
          jump    = np.sqrt(indicater.integrate(domain, geom, degree, ns.jump, domain.interfaces, interfaces=True))
    
          z_int     = np.sqrt(indicater.integrate(domain, geom, degree, ns.zsharp, domain))
          s_int     = np.sqrt(indicater.integrate(domain, geom, degree, ns.ssharp, domain))
          z_outflow = np.sqrt(indicater.integrate(domain, geom, degree, ns.zsharp, domain.boundary['right']))
          z_jump    = np.sqrt(indicater.integrate(domain, geom, degree, ns.zsharp, domain.interfaces, interfaces=True))
          ### Get indicaters ###
    
    
          ### Refine mesh ###
          if method == 'goaloriented':

              # assemble indicaters
              inter = incom*s_int + force*z_int
              iface = jump*z_jump
              bound = outflow*z_outflow

              indicators =  inter + iface + bound

              plotter.plot_indicators('contributions_'+method+'_'+str(nref), domain, geom, {'momentum':force,'dualvelocity':z_int,'interface':jump,'dualjump':z_jump,'incompressibility':incom,'dualpressure':s_int,'outflow':outflow,'dualoutflow':z_outflow}, normalize=False, alpha=.5)
              plotter.plot_indicators('indicators_'+method+'_'+str(nref), domain, geom, {'indicator':indicators}, normalize=False, alpha=.5)

              domain, refined = refiner.refine(domain, indicators, num, evalbasis, maxlevel=maxrefine+uref-1, select_type='same_level')

          if method == 'residualbased':

              # assemble indicaters
              inter = (incom + force)*h
              iface = jump*np.sqrt(h)
              bound = outflow*np.sqrt(h)

              indicators =  inter + iface + bound

              plotter.plot_indicators('indicators_'+method+'_'+str(nref), domain, geom, {'indicator':indicators,'incompressibility':incom*h,'momentum':force*h,'interfaces':iface,'outflow':bound}, normalize=False, alpha=.5)
              
              domain, refined = refiner.refine(domain, indicators, num, evalbasis, maxlevel=maxrefine+uref-1, select_type='same_level')

          if method == 'uniform':

              domain = domain.refine(1)
              refined = True

              if nref == maxuref:
                  refined = False                    

          # Stop with the refinement loop if nothing is refined
          if not refined:
              break

        plotter.plot_mesh('mesh'+method+str(M1),domain,geom, title=method+str(M1))
        plotter.plot_solution('pressure'+method+str(M1),domain,geom,ns.p)
        plotter.plot_solution('dualpressure'+method+str(M1),domain,geom,ns.s)
        plotter.plot_streamlines('velocity'+method+str(M1),domain,geom,ns,ns.u)
        plotter.plot_streamlines('dualvelocity'+method+str(M1),domain,geom,ns,ns.z)
    
      plotter.plot_convergence('Estimated_error_force_'+str(M1),ndofs,error_force,labels=['dofs','Estimated error'],slopemarker=True)
      plotter.plot_convergence('Estimated_error_incomp_'+str(M1),ndofs,error_incomp,labels=['dofs','Estimated error'],slopemarker=True)
      plotter.plot_convergence('Estimated_error_QoI'+str(M1),ndofs,error_qoi,labels=['dofs','Estimated error'])

with config(verbose=3,nprocs=6):
    cli.run(main)


