import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .magblock import Block, Parallelepiped, Polyhedron


#todo: incluir transformacoes no plot

#todo: blockbox para configurar automaticamente limites
def plot_magnetic_block(mblock:Block,vertices=False,
                        ax=None,show=True,**kwargs):

    if not ax:
        ax = plt.subplot(projection='3d')

    #? deixar sempre aspect='equal' ?
    settings = {setting: kwargs.pop(setting)
                for setting in ['xlim','ylim','zlim','title',
                                'xlabel','ylabel','zlabel']
                    if setting in kwargs}
    ax.set(aspect='equal',**settings)


    if isinstance(mblock,Parallelepiped):
        faces = mblock.faces()
        vs =  mblock.vertices()

    elif isinstance(mblock,Polyhedron):
        faces = [mblock.face(sigma)
                    for sigma in range(len(mblock.faces_indices))]
        vs = np.array(mblock.vertices)


    if vertices:
        ax.plot(vs[:,0],vs[:,1],vs[:,2],'.',c='k')
    else:
        ax.add_collection3d(Poly3DCollection(faces,**kwargs))
    
    if show:
        plt.show()





