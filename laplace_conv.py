import numpy
from utilities import writer
from nutils import *
import matplotlib.pyplot as plt
from utilities import plotter

import treelog

#def main(case = 'square'):
def main(case = 'lshape',):

    #folder = '4dec_degree2/'
    #folder = '12dec_topadded/'
    folder = ''

    datalog = treelog.DataLog('../results/laplace/'+folder+'images')

    methods = ['residual','goal','uniform']
    poitypes = ['center','corner']

    ####
    #poitypes = ['corner']
    #methods = ['uniform']
    ####

    for poitype in poitypes: 

        #for error in ['error_exact','error_qoi','error_est','sum_residual','sum_goal']:
        for error in ['norm_L2','norm_H1','residual_e','sum_ind','error_qoi','residual_z','sum_goal']:
            
            xval  = {}
            yval  = {}
            level = {}

            for i, method in enumerate(methods):

                text = writer.read('../results/laplace/'+folder+case+methods[i]+poitype)
        
                xval[method]  = text['ndofs']      
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
            ####
            slopemarker = None
            ####

            labels = ['amount of elements','error']

            #with treelog.add(datalog):
            plotter.plot_convergence(poitype+'-'+error, xval, yval, slopemarker=slopemarker, levels=level, labels=['Amount of dofs','Error'])
        
cli.run(main)
