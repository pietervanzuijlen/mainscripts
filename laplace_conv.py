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
    methods = ['residual','goal']
    poitypes = ['center','corner']

    for poitype in poitypes: 

        for error in ['norm_L2','norm_H1','residual_e','sum_ind','error_qoi','residual_z','sum_goal']:
            
            xval  = {}
            yval  = {}
            level = {}

            for i, method in enumerate(methods):

                text = writer.read('../results/laplace/'+folder+case+methods[i]+poitype)
        
                xval[method]  = numpy.sqrt(text['ndofs'])
                yval[method]  = text[error]      
                level[method] = text['maxlvl']

####
#            if error == 'error_exact' or error == 'error_qoi':
#                slopemarker = {}
#                slopemarker['uniform'] = [(-2,3),.03]
#                slopemarker['goal']    = [(-3,2),.12]
#                #slopemarker['goal']    = [(-3,2),.09]
#            else:
#                slopemarker = None
####
            slopemarker = None

            labels = ['sqrt(ndofs)','error']

            #with treelog.add(datalog):
            plotter.plot_convergence(poitype+'-'+error, xval, yval, slopemarker=slopemarker, levels=level, labels=labels)
        
cli.run(main)
