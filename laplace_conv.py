import numpy
from utilities import writer
from nutils import *
import matplotlib.pyplot as plt
from utilities import plotter

import treelog

#def main(case = 'square'):
def main(case = 'lshape',):

    datalog = treelog.DataLog('../results/laplace/'+'images')

    methods = ['residual','goal','uniform']
    poitypes = ['center','corner']
    poitypes = ['singularity']

    for poitype in poitypes: 

        for error in ['error_exact','error_est','error_qoi','residual_z','nelems']:
            
            xval  = {}
            yval  = {}
            level = {}

            for i, method in enumerate(methods):

                text = writer.read('../results/laplace/'+case+methods[i]+poitype)
        
                xval[method]  = numpy.sqrt(text['ndofs'])
                yval[method]  = text[error]      
                level[method] = text['maxlvl']

            if error == 'error_est':
                slopemarker = {}
                slopemarker['uniform'] = [(-1,1),.03]
                slopemarker['goal']    = [(-2,1),.12]
                #slopemarker['goal']    = [(-3,2),.09]
            else:
                slopemarker = None

            labels = ['sqrt(ndofs)','error']

            #with treelog.add(datalog):
            plotter.plot_convergence(poitype+'-'+error, xval, yval, slopemarker=slopemarker, levels=level, labels=labels)
        
cli.run(main)
