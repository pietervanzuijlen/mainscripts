# -*- coding: utf-8 -*-
"""
**Example of Stokes equation through porous domain:**

This example demonstrates the skeleton-stabilized immersogeometric simulation
of Stokes equation through porous domain. The porous domain is either created
artificially using `create_artificial_data`,  which can be connected
or disconnected domain or scanned from the existing voxel data. The essential boundary
conditions are imposed weakly using Nitsche's method on Dirichlet boundary(Γ_D).
Neumann condition is imposed at the inflow boundary (Γ_N) of the domain. The pressure
stabilization is implemented on the skeleton of the background mesh and Ghost penalty 
stabilization of the velocity components is implemented at the cut boundary. The 
boundary value problem is ::

  - ∇.(2μ ∇^s u) + ∇ p = 0    in Ω
                ∇ . u  = 0    in Ω
                    u  = 0    on Γ_D
    2μ ∇^s u . n - p n = 1    on Γ_N

"""

#!/usr/bin/env python3

from nutils import cli, log, mesh, function, topology, solver, cache, numeric, matrix, transform, export
import numpy, itertools, pathlib, unittest, numpy.linalg, json
from immersogeometric import voxel, util, trimming

# data directory
datadir = pathlib.Path(__file__).resolve().parent / 'data'


def main(voxeldata               ,
         mu                      ,
         beta                    ,
         gammaskeleton           ,
         gammaghost              ,
         pbar                    ,
         nelems                  ,
         degree                  ,
         maxrefine               ,
         threshold    = 0        ,
         compress     = True     ,
         caching      = False    ,
         plotting     = True     ):

  """`main` functions with different parameters

  Parameters
  ----------
  voxeldata : cli.path(filename.json)
      voxel data object
  mu : float
      fluid viscosity
  beta : float
      nitsche parameter
  gammaskeleton : float
      skeleton parameter
  gammaghost : float
      ghost parameter
  pbar : float
      pressure drop
  nelems : (int, int)
      number of elements
  degree : int
      discretizatio ndegree
  maxrefine : int
      bisectioning steps
  threshold : 
      threshold value
  compress : bool
      compressed integration
  caching : bool
      disable/enable caching
  plotting : bool
      diable/enable plotting
  
  Returns
  -------
  lhs0  : <Array> 
      solution u
  phi   : float
      domain porosity
  kappa : float
      effective permeability

  """

  # set up the cache
  fcache = cache.FileCache(voxeldata,nelems,degree,maxrefine,threshold) if caching else lambda func, *args, **kwargs : func(*args, **kwargs)

  with log.context('input'):

    # print data information
    voxeldata.log()

    # unpack the topology and the geometry function
    voxeltopo, voxelgeom = voxeldata.mesh

    # log the threshold and porosity
    porosity = voxeldata.count_porous_voxels(threshold)/voxeldata.count_voxels()
    log.user('voxel data threshold: {}'.format(threshold))
    log.user('voxel data porosity : {}'.format(porosity))

    # smoothen the levelset
    levelset = voxeltopo.projection(threshold-voxeldata.func(eval_topo=voxeltopo,eval_unit_geom=voxeldata.unit_geom), onto=voxeltopo.basis('spline',degree=degree), geometry=voxelgeom, ptype='convolute', ischeme='gauss{}'.format(degree))

    # post-processing
    if plotting:
      vbezier = voxeltopo.sample('bezier', 2**maxrefine+1)
      points, voxelvalues, smoothvalues = vbezier.eval([voxelgeom,threshold-voxeldata.func(eval_topo=voxeltopo,eval_unit_geom=voxeldata.unit_geom,discontinuous=True),levelset])
      export.vtk('voxels', vbezier.tri, points, intensity=voxelvalues, smooth=smoothvalues)

    # create the ambient domain grid
    topo, geom = mesh.rectilinear( [numpy.linspace(0,length,nelem+1) for nelem,length in zip(nelems,voxeldata.lengths)] )

    #Trim the topology by wrapping the levelset function
    levelset = util.TopoMap(levelset, func_topo=voxeltopo, eval_topo=topo, eval_unit_geom=geom/voxeldata.lengths)

    # recalibrate the threshold
    uniform = topo.sample('uniform', 2**(maxrefine+2))
    intensities = uniform.eval( levelset )
    threshold = voxel.get_threshold(intensities, 1-porosity)
    levelset = levelset-threshold
    log.user('level set threshold shift: {}'.format(threshold))

  @fcache
  def trimmed_topo():
    return topo.trim(levelset, maxrefine=maxrefine)

  # filter out parts disconnected from the Neumann boundary (in this case left & right)
  ttopo = trimming.filter_disconnected(trimmed_topo.boundary['left,right'], trimmed_topo)
  ftopo = trimmed_topo-ttopo

  # retieve the background domain
  btopo = topology.SubsetTopology(topo, [elem.reference  if transform.lookup(elem.transform,ttopo.edict) else elem.reference.empty for elem in topo])

  # retrieve the skeleton mesh
  skeleton = btopo.boundary
  skeleton = btopo.interfaces

  ghost = []
  for iface in skeleton:
    for trans in iface.transform, iface.opposite:

      # find the corresponding element in the background mesh
      ielemb, tailb = transform.lookup_item(trans,btopo.edict)
      belem = btopo.elements[ielemb]

      # find the corresponding element in the trimmed mesh
      ielemt, tailt = transform.lookup_item(trans,ttopo.edict)
      telem = ttopo.elements[ielemt]

      if belem != telem:
        assert belem.transform == telem.transform
        assert belem.opposite  == telem.opposite
        ghost.append(iface)
        break

  ghost = topology.UnstructuredTopology(btopo.ndims-1, ghost)

  # namespace initialization
  ns = function.Namespace()
  ns.mu   = mu
  ns.h    = numpy.linalg.norm(numpy.array(voxeldata.lengths)/nelems)
  ns.beta = beta
  ns.gammaskeleton = gammaskeleton
  ns.gammaghost = gammaghost
  ns.x    = geom
  ns.pbar = pbar

  # construct bases
  ns.ubasis, ns.pbasis = function.chain([
     ttopo.basis( 'spline', degree=degree ).vector(ttopo.ndims),
     ttopo.basis( 'spline', degree=degree )])

  ns.jumpubasis = function.jump(ns.ubasis)
  ns.jumppbasis = function.jump(ns.pbasis)

  # formulation
  ns.u_i = 'ubasis_ni ?lhs_n'
  ns.p   = 'pbasis_n ?lhs_n'

  ns.jumpu_i = 'jumpubasis_ni ?lhs_n'
  ns.jumpp   = 'jumppbasis_n ?lhs_n'

  # volume terms
  ns.volume_n = 'mu ubasis_ni,j (u_i,j + u_j,i) - ubasis_nk,k p - u_k,k pbasis_n'

  # dirichlet boundary
  ns.nitsche_n  = '-mu ( (u_i,j + u_j,i) n_i ubasis_nj + (ubasis_ni,j + ubasis_nj,i) n_i u_j ) ' \
                  '+ mu (beta  / h) ubasis_ni u_i ' \
                  '+ p ubasis_ni n_i + pbasis_n u_i n_i'

  # inflow boundary
  ns.inflow_n = 'pbar n_i ubasis_ni'

  # skeleton stabilization
  ns.skeleton_n = '-gammaskeleton h^{} {} {}'.format(2*degree+1, normal_derivative('jumppbasis_n', 'abc'[:degree]), normal_derivative('jumpp_', 'efg'[:degree]))

  # ghost stabilization
  ns.ghost_n = 'gammaghost h^{} {} {}'.format(2*degree-1, normal_derivative('jumpubasis_ni', 'abc'[:degree]), normal_derivative('jumpu_i', 'def'[:degree]))

  # nitsche topology
  ntopo = ttopo.boundary['trimmed,top,bottom' if ttopo.ndims==2 else 'trimmed,top,bottom,front,back'].simplex

  # patch for compressed integration of trimmed simplices
  if compress:
    ntopo = trimming.enable_compressed_integration(ntopo)

  # assembly
  res  = ttopo.integral('volume_n d:x' @ ns, degree=2*(degree+1))
  res += ntopo.integral('nitsche_n d:x' @ ns, degree=2*(degree+1))
  res += ttopo.boundary['left'].integral('inflow_n d:x' @ ns, degree=2*(degree+1))
  res += skeleton.integral('skeleton_n d:x' @ ns, degree=2*(degree+1))
  res += ghost.integral('ghost_n d:x' @ ns, degree=2*(degree+1))

  lhs0 = solver.solve_linear('lhs', res)
  ns = ns(lhs=lhs0)

  ###################
  # post-processing #
  ###################

  with log.context('output'):

    # compute geometry properties
    trimmed_volume = ttopo.integrate(function.J(geom), ischeme='gauss1')
    removed_volume = ftopo.integrate(function.J(geom), ischeme='gauss1') if len(ftopo) > 0 else 0
    areaeff = topo.boundary['left'].integrate(function.J(geom), ischeme='gauss1')
    phi = (trimmed_volume+removed_volume)/voxeldata.volume
    log.user('trimmed domain porosity: {}'.format(trimmed_volume/voxeldata.volume))
    log.user('total domain porosity: {} (voxel input: {})'.format(phi,porosity))
    log.user( 'effective (in/out)flow area: {}'.format(areaeff) )

    # compute the averaged in and out flow conditions
    areain , qin , pin  = ttopo.boundary['left'] .integrate([function.J(geom), '-u_i n_i d:x' @ ns, ns.p*function.J(geom)], degree=(degree+1) )
    areaout, qout, pout = ttopo.boundary['right'].integrate([function.J(geom), 'u_i n_i d:x'  @ ns, ns.p*function.J(geom)], degree=(degree+1) )
    log.user( '(in/out)flow area: {} / {}'.format(areain,areaout))
    log.user( '(in/out)flow pressure average: {} / {}'.format(pin/areain,pout/areaout))
    log.user( '(in/out)flow flux: {} / {}'.format(qin,qout) )

    # compute the effective permeability
    kappa = -mu*((qin+qout)/2)/(areaeff*(pout/areaout-pin/areain))
    log.user( 'effective premeability: {}'.format(kappa))

    # vtk output
    if plotting:

      # disconnected domain
      if len(ftopo) > 0:
        fbezier = ftopo.simplex.sample('bezier', 2**maxrefine+1)
        points  = fbezier.eval(ns.x)
        export.vtk('removed', fbezier.tri, points)

      # trimmed domain
      tbezier = ttopo.simplex.sample('bezier', 2**maxrefine+1)
      points, pressures, velocities = tbezier.eval([ns.x, ns.p, ns.u])
      export.vtk('domain', tbezier.tri, points, p=pressures, u=velocities)

      # background mesh
      bbezier = btopo.sample('bezier', 2**maxrefine+1)
      points  = bbezier.eval(geom)
      export.vtk('background', bbezier.tri, points)

      # skeleton
      sbezier = skeleton.sample('bezier', 2**maxrefine+1)
      points  = sbezier.eval(geom)
     # export.vtk('skeleton', sbezier.tri, points) #TODO issue: generalize export.vtk #322

      # ghost
      gbezier = ghost.sample('bezier',2**maxrefine+1)
      points  = gbezier.eval(geom)
     # export.vtk('ghost', gbezier.tri, points) #TODO issue: generalize export.vtk #322

  # return solution vector and effective permeability
  return lhs0, phi, kappa

# creates normal derivative strings for name spaces
def normal_derivative( function, derivative_indices ):
  return '{},{} {}'.format(function, derivative_indices, ' '.join(['n_{}'.format(derivative_index) for derivative_index in derivative_indices]))

def scan(name          = 'connected_2d',
         mu            = 1             ,
         beta          = 100           ,
         gammaskeleton = 0.05          ,
         gammaghost    = 0.0005        ,
         pbar          = 1.            ,
         nelems        = 8             ,
         degree        = 2             ,
         maxrefine     = 2             ,
         ncoarsegrain  = 0             ,
         compress      = True          ,
         select        = ''            ,
         caching       = False         ,
         plotting      = True          ):

  """`scan` functions with different parameters

  parameters
  ----------
  name         : string
      voxel data set name
  mu           : float
      fluid viscosity
  beta         : float
      nitsche parameter
  gammaskeleton : float
      skeleton parameter
  gammaghost   : float
      ghost parameter
  pbar         : float
      pressure drop
  nelems       : (int, int)
      number of elements
  degree       : int
      discretizatio ndegree
  maxrefine    : int
      bisectioning steps
  ncoarsegrain : int
      Coarsegraining integration
  compress     : bool
      compressed integration
  select       : str 
      selection "start1:stop1,start2:stop2,..." means (slice(start1,stop1),slice(start2,stop2),...)
  caching      : bool
      disable/enable caching
  plotting     : bool
      diable/enable plotting

  Returns
  -------
  main : function
  """

  # set the json file name
  fjson = datadir / '{}.json'.format(name)
  assert fjson.is_file(), '{} not found in data directory; can be generated using create_artificial_data'.format(fjson.name)

  # load the voxel data file
  voxeldata, props = voxel.jsonread(fjson)

  # slice the voxel data: string "i,j" means slice(i,j)
  if select != '':
    if ',' in select:
      voxeldata = voxeldata[tuple(slice(*map(int,s.split(':'))) for s in select.split(','))]
    else:
      voxeldata = voxeldata[slice(*map(int,select.split(':')))]

  # coarse grain the voxel data file
  voxeldata = voxeldata.coarsegrain(ncoarsegrain=ncoarsegrain)

  # general input data formatting
  threshold = voxeldata.get_threshold(props['porosity']) if 'porosity' in props else 0
  nelems    = (nelems,)*voxeldata.ndims

  # call the main function
  return main(voxeldata    = voxeldata    , 
              mu           = mu           ,
              beta         = beta         ,
              gammaskeleton= gammaskeleton,
              gammaghost   = gammaghost   ,
              pbar         = pbar         ,
              nelems       = nelems       ,
              degree       = degree       ,
              maxrefine    = maxrefine    ,
              threshold    = threshold    ,
              compress     = compress     ,
              caching      = caching      ,
              plotting     = plotting      )

def create_artificial_data(case   = 'connected',
                           ndims  = 2          ,
                           R      = 0.5        ,
                           L      = 2.5        , 
                           nvox   = 40         ,
                           nint   = 8          ,
                           dtype  = '<i2'      ,
                           name   = None       ):

  """Creates artificial voxel data

  Parameters
  ----------
  case  : string
      geometry case
  ndims : int
      number of dimensions
  R     : float
      radius 
  L     : float
      domain size
  nvox  : int
      Number of voxels per direction
  nint  : int
      number for integration points
  dtype : i<int> 
      gray scale data type
  name  : bolean
      data set name
  """

  #set the file name for the raw data
  fraw = datadir / '{}_{}d.raw'.format(case,ndims) if not name else name

  #construct the voxel topology
  topo, geom = mesh.rectilinear([numpy.linspace(0,L,nvox+1)]*ndims)

  #construct the levelset function
  assert R < L/2
  if case == 'connected':
    lvl = R-function.norm2(geom-L/2)
    for xvert in itertools.product([0,L],repeat=ndims):
      lvl = function.max(R - function.norm2(geom-xvert), lvl)
  elif case == 'disconnected':
    rtorus  = function.sqrt((function.abs(L-geom[-1]))**2+(L/2-function.norm2(function.abs(L/2-geom[:-1])))**2)
    rbubble = function.sqrt((function.abs(geom[-1]))**2+((function.abs(L/2-geom[:-1])**2).sum()))
    lvl     = function.max(function.min(L/2-geom[-1],rbubble-R),(L/2-R)-rtorus)
  else:
    raise RuntimeError('Unknown case "{}"'.format(case))

  #compute and save the volume fractions
  data = topo.elem_mean(function.heaviside(lvl), geometry=geom, ischeme='uniform%s'%nint)

  #rescale the data to the dtype range
  nf   = numpy.iinfo(dtype)
  data = (nf.max-nf.min)*data+nf.min

  #save the raw data
  data.astype(dtype).tofile(str(fraw))

  #write the json file
  with fraw.with_suffix('.json').open('w') as fout:
    json.dump({'FNAME' : str(fraw.name),
               'DIMS'  : [nvox]*ndims  ,
               'DTYPE' : dtype         ,
               'ORDER' : 'C'           ,
               'SIZE'  : [L/nvox]*ndims }, fout)


################
# Unit testing #
################

class test(unittest.TestCase):

  def test_connected_2d(self):
    # create data`
    create_artificial_data(case='connected', ndims=2)

    # run simulation
    lhs, phi, kappa = scan(name         = 'connected_2d',
                           nelems       = 8             ,
                           degree       = 2             ,
                           maxrefine    = 2             ,
                           ncoarsegrain = 0             ,
                           caching      = False         ,
                           plotting     = False          )

    # porosity
    numpy.testing.assert_almost_equal(phi, 0.73922781646251678, decimal=10)

    # permeability
    numpy.testing.assert_almost_equal(kappa, 0.015436515338062966, decimal=10)

    # solution vector
    numeric.assert_allclose64(lhs,'eNptjj9IW0Ecx0GQdBRBgkNDkCfEXC6/e3/uTokUS9EHAemgToKCINUhgzgUsSAZugiZHAVJQAvFPaCCXURF9PlyufjnRVR0MG5BQRoE7W8u4QPH9z5w3Kd+bMRzzEVyzIjXj59IylsORhhBRthykPKeyMJl6jwP39gu7OKZh9T5wuWcEYEC62BWzIp1sAKLwJyxGa3CLcvp4CQ4yelbVoXNaDPX7G2zP5q11P/rzXhD9ODsTkVoX8+2WjEbiUMzq/+WpvQa+ZBcIr1eqNwadOotP+1flYxkEVx4IK/dLUZMDfp7fkOHSD/swAV9qxZuwjqtSGXVnPYq8SJ9JJPRyegjKdJKfNpbNUklrcK6cPNWvaA70A8h0tB7/qAfUy3Ga/cDcaEIRvKqlPa3/E7dGoTKvd4SVqyRKSzK6kOsWzG3VV9PhN6pg7MhmvHeRUIOyM/IgEzId/FdlEWXdOUY4souWUYzL/LiWYD8ioB8xtu8GBMZ8VvURLtslzVcGTQfhSE+iXHxAxnHZaA55fv8F1/kX5BFXPv8lP/kE/zeCTumbdph596ZQDPKX5ya3WbN0lnaZtXsF2eUD/MZZ926hqzKqmtYt2acYb5h29YR+4McMdvasP8Bwm70ww==')

  def test_disconnected_2d(self):
    # create data`
    create_artificial_data(case='disconnected', ndims=2)

    # run simulation
    lhs, phi, kappa = scan(name         = 'disconnected_2d',
                           nelems       = 8                ,
                           degree       = 2                ,
                           maxrefine    = 2                ,
                           ncoarsegrain = 0                ,
                           caching      = False            ,
                           plotting     = False             )

    # porosity
    numpy.testing.assert_almost_equal(phi, 0.42860529944300652, decimal=10)

    # permeability
    numpy.testing.assert_almost_equal(kappa, 0.0044653799861275167, decimal=10)

    # solution vector
    numeric.assert_allclose64(lhs,'eNpVjU8og3EcxnEi4UBa5jCWtvfv7+V9v9+htZNdlJajRaRcpFZyUCtKxk0p5EK0mVqKk/McXHi39+9eW8gOhoss20E58J5kt+fz9PR5ireLyrXQJjRn/PKcciVESH12IPMsB9UPUiGbep2RoEuZqjpIzskqN/cQKaSyKW2DD5Fh/oAi7lqqXdZa/vuLf69OtU+f5sOaJmTUIv1p9PMryqXsZbwmx1L8s3mqhj2WnjBnmBcuyu2xb4+HBm15aJ1pYd+oBfcsa1GFHG9WDIe180Bzk8yR2aXH9HnD0Zs384xh3GndzARJ5kXiyrWyirYmjIolcs8da0meKGXpCad8Qd8SnkAaO3xfOIUxCXEd9zGKFG5DEEI4igE7N+EdnIEL2+3UgN9QhleQ4R5K8A5V+IFG3IJdSELabjpxCC9AgiYYsXnMdgUwLvWIWXEZbjCOfsmpuoVx+AVfUa0k')

if __name__ == '__main__':
  cli.choose(create_artificial_data, scan)
