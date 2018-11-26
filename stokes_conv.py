import numpy
from utilities import writer
from nutils import *
import matplotlib.pyplot as plt

#def main(case = 'square'):


def main(case = 'stokes',):


    for M1 in log.iter('Central circle position', [0.5,0.4,0.35]):

        methods = ['residual','goal','uniform']
        poitype = 'corner'
             
       
        color = ['c','b','y','g','y']
        marker= [':*','-+']
        
        unif = []
        goal = []
        error= []
    
        text = writer.read('results/'+case+methods[0]+str(M1))
    
        resid_residual     = text['residual'] 
        resid_sum_residual = text['sum_residual'] 
        resid_sum_goal     = text['sum_goal']    
        resid_nelems       = text['nelems']      
    
        text = writer.read('results/'+case+methods[1]+str(M1))
    
        goal_residual     = text['residual'] 
        goal_sum_residual = text['sum_residual'] 
        goal_sum_goal     = text['sum_goal']    
        goal_nelems       = text['nelems']      
    
        text = writer.read('results/'+case+methods[2]+str(M1))
    
        unif_residual     = text['residual'] 
        unif_sum_residual = text['sum_residual'] 
        unif_sum_goal     = text['sum_goal']    
        unif_nelems       = text['nelems']      
    
        
        with export.mplfigure('sum_residual_indicators.png') as fig:
            ax = fig.add_subplot(111)
            im = ax.loglog(unif_nelems, unif_sum_residual, color[0])
            im = ax.loglog(resid_nelems, resid_sum_residual, color[1])
            im = ax.loglog(goal_nelems, goal_sum_residual, color[2])
            ax.autoscale(enable=True, axis='both', tight=True)
            ax.set_xlabel('# dofs')
            ax.set_ylabel('error')
            ax.legend(['uniform','residual-based','goal-oriented'])
            ax.set_title('M1 '+str(M1) + ' | Sum residual indicators')
    
        with export.mplfigure('sum_goal_indicators.png') as fig:
            ax = fig.add_subplot(111)
            im = ax.loglog(unif_nelems, unif_sum_goal, color[0])
            im = ax.loglog(resid_nelems, resid_sum_goal, color[1])
            im = ax.loglog(goal_nelems, goal_sum_goal, color[2])
            ax.autoscale(enable=True, axis='both', tight=True)
            ax.set_xlabel('# dofs')
            ax.set_ylabel('error')
            ax.legend(['uniform','residual-based','goal-oriented'])
            ax.set_title('M1 '+str(M1) + ' | Sum goal indicators')
    
        with export.mplfigure('residual.png') as fig:
            ax = fig.add_subplot(111)
            im = ax.loglog(unif_nelems, unif_residual, color[0])
            im = ax.loglog(resid_nelems, resid_residual, color[1])
            im = ax.loglog(goal_nelems, goal_residual, color[2])
            ax.autoscale(enable=True, axis='both', tight=True)
            ax.set_xlabel('# dofs')
            ax.set_ylabel('error')
            ax.legend(['uniform','residual-based','goal-oriented'])
            ax.set_title('M1 '+str(M1) + ' | residual')

#    with export.mplfigure('sum_residual_indicators.png') as fig:
#        ax = fig.add_subplot(111)
#        im = ax.loglog(unif_nelems, unif_sum_residual, color[0])
#        im = ax.loglog(goal_nelems, goal_sum_residual, color[1])
#        im = ax.loglog(resid_nelems, resid_sum_residual, color[2])
#        ax.autoscale(enable=True, axis='both', tight=True)
#        ax.set_xlabel('Refinement')
#        ax.set_ylabel('Sum of residual-based indicators')
#        ax.legend(['uniform','goal-oriented','residual-based'])
#
#    with export.mplfigure('sum_goal_indicators.png') as fig:
#        ax = fig.add_subplot(111)
#        im = ax.loglog(unif_nelems, unif_sum_goal, color[0])
#        im = ax.loglog(goal_nelems, goal_sum_goal, color[1])
#        im = ax.loglog(resid_nelems, resid_sum_goal, color[2])
#        ax.autoscale(enable=True, axis='both', tight=True)
#        ax.set_xlabel('Refinement')
#        ax.set_ylabel('Sum of goal-oriented indicators')
#        ax.legend(['uniform','goal-oriented','residual-based'])


cli.run(main)
