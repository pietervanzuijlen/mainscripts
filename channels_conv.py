import numpy
from utilities import writer
from nutils import *
import matplotlib.pyplot as plt
from utilities import plotter

import treelog

def main(case = 'channels',):

    datalog = treelog.DataLog('../results/channels/images')

    methods = ['residual','goal','uniform']

    for error in ['residual','sum_residual','sum_goal']:
            
        xval  = {}
        yval  = {}
        level = {}

        for i, method in enumerate(methods):

            text = writer.read('../results/channels/'+case+methods[i])
    
            xval[method]  = text['nelems']      
            yval[method]  = text[error]      
            level[method] = text['maxlvl']

        slopemarker = {}
        if error == 'residual':
            slopemarker['uniform'] = [(-1,4),.03]
        elif error == 'sum_residual':
            slopemarker['uniform'] = [(-4,3),.06]
        elif error == 'sum_goal':
            slopemarker['uniform'] = [(-1,1),.03]

        labels = ['amount of elements','error']

        with treelog.add(datalog):
            plotter.plot_convergence(error, xval, yval, slopemarker=slopemarker, levels=level)
        
cli.run(main)
