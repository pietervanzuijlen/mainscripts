import numpy
from utilities import writer
from nutils import *
import matplotlib.pyplot as plt

#def main(case = 'square'):
def main(case = 'lshape',):


    methods = ['residual','goal','uniform']
    poitype = 'corner'
         
   
    color = ['c','b','y','g','y']
    marker= [':*','-+']
    
    unif = []
    goal = []
    error= []

    text = writer.read('../results/'+case+methods[0]+poitype)

    resid_error_exact = text['error_exact']   
    resid_error_qoi   = text['error_qoi']   
    resid_error_est   = text['error_est']   
    resid_sum_residual= text['sum_residual'] 
    resid_sum_goal    = text['sum_goal']    
    resid_nelems      = text['nelems']      

    text = writer.read('../results/'+case+methods[1]+poitype)

    goal_error_exact = text['error_exact']   
    goal_error_qoi   = text['error_qoi']   
    goal_error_est   = text['error_est']   
    goal_sum_residual= text['sum_residual'] 
    goal_sum_goal    = text['sum_goal']    
    goal_nelems      = text['nelems']      

    text = writer.read('../results/'+case+methods[2]+poitype)

    unif_error_exact = text['error_exact']   
    unif_error_qoi   = text['error_qoi']   
    unif_error_est   = text['error_est']   
    unif_sum_residual= text['sum_residual'] 
    unif_sum_goal    = text['sum_goal']    
    unif_nelems      = text['nelems']      

    
    with export.mplfigure('exact_error.jpg') as fig:
        ax = fig.add_subplot(111)
        im = ax.loglog(unif_nelems, unif_error_exact, color[0])
        im = ax.loglog(goal_nelems, goal_error_exact, color[1])
        im = ax.loglog(resid_nelems, resid_error_exact, color[2])
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_xlabel('DOF`s')
        ax.set_ylabel('Exact error')
        ax.legend(['uniform','goal-oriented','residual-based'])

    with export.mplfigure('quantity_of_interest.jpg') as fig:
        ax = fig.add_subplot(111)
        im = ax.loglog(unif_nelems, unif_error_qoi, color[0])
        im = ax.loglog(goal_nelems, goal_error_qoi, color[1])
        im = ax.loglog(resid_nelems, resid_error_qoi, color[2])
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_xlabel('Refinement')
        ax.set_ylabel('Error in quantity of interest')
        ax.legend(['uniform','goal-oriented','residual-based'])

    with export.mplfigure('estimated_error.jpg') as fig:
        ax = fig.add_subplot(111)
        im = ax.loglog(unif_nelems, unif_error_est, color[0])
        im = ax.loglog(goal_nelems, goal_error_est, color[1])
        im = ax.loglog(resid_nelems, resid_error_est, color[2])
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_xlabel('Refinement')
        ax.set_ylabel('Error estimate')
        ax.legend(['uniform','goal-oriented','residual-based'])

    with export.mplfigure('sum_residual_indicators.jpg') as fig:
        ax = fig.add_subplot(111)
        im = ax.loglog(unif_nelems, unif_sum_residual, color[0])
        im = ax.loglog(goal_nelems, goal_sum_residual, color[1])
        im = ax.loglog(resid_nelems, resid_sum_residual, color[2])
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_xlabel('Refinement')
        ax.set_ylabel('Sum of residual-based indicators')
        ax.legend(['uniform','goal-oriented','residual-based'])

    with export.mplfigure('sum_goal_indicators.jpg') as fig:
        ax = fig.add_subplot(111)
        im = ax.loglog(unif_nelems, unif_sum_goal, color[0])
        im = ax.loglog(goal_nelems, goal_sum_goal, color[1])
        im = ax.loglog(resid_nelems, resid_sum_goal, color[2])
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_xlabel('Refinement')
        ax.set_ylabel('Sum of goal-oriented indicators')
        ax.legend(['uniform','goal-oriented','residual-based'])


cli.run(main)
