
import numpy as np




class Block:
    """Magnetic base block."""

    def __init__(self,magnetization=None):
        """
        A uniformly magnetized block.

        Parameters
        ----------
        magnetization : ndarray, shape (3,)
            The magnetization vector of the block [T]; (mx, my, mz).
        """
        self._m = np.asarray(magnetization) if magnetization else np.zeros(3)

        # self._v = np.array([0,0,0]) # translation vector
        self._T = np.eye(3)         # transformation matrix #todo: 4D

    @property
    def magnetization(self):
        """Magnetization vector of the rectangular block [T]; (mx, my, mz)"""
        return self._m
    
    @magnetization.setter
    def magnetization(self,magnetization):
        self._m = np.asarray(magnetization)

    # @property
    # def translation(self):
    #     """Translation vector [m]; (vx, vy, vz)"""
    #     return self._v

    @property
    def transformation(self):
        """Resultant Transformation Matrix"""
        return self._T

    def isPointInsideBlock(self,r0) -> bool:
        raise NotImplementedError

    #todo: print warning of point on surface (face, edge, vertex)
    def Q(self,r0):
        raise NotImplementedError
    
    def Mfield(self,r0):
        """
        Compute the magnetization field of the block at position r0.

        Parameters
        ----------
        r0 : ndarray, shape (3,)
            Position at which to compute the magnetic field.

        Returns
        -------
        M : ndarray, shape (3,1)
            The magnetization contribution to the magnetic field at position r0, in Tesla.

        """
        T = self.transformation
        Tinv = np.linalg.inv(T)
        m = self.magnetization
        r0 = np.asarray(r0)

        if self.isPointInsideBlock(Tinv @ r0):
            return T @ m
        else:
            return np.zeros(3)

    def Hfield(self,r0):
        """
        Compute the magnetic field H of the block at position r0.

        Parameters
        ----------
        r0 : ndarray, shape (3,)
            Position at which to compute the magnetic field.

        Returns
        -------
        H : ndarray, shape (3,)
            The magnetic field H at position r0 [T].
            
        """
        T = self.transformation
        Tinv = np.linalg.inv(T)
        m = self.magnetization
        r0 = np.asarray(r0)

        return T @ (self.Q(Tinv @ r0) @ m)
    
    def Bfield(self,r0):
        """
        Compute the magnetic field B of the block at position r0.

        Parameters
        ----------
        r0 : ndarray, shape (3,)
            Position at which to compute the magnetic field.

        Returns
        -------
        B : ndarray, shape (3,)
            The magnetic field B at position r0 [T].

        """
        return self.Hfield(r0) + self.Mfield(r0)

    # primeiro considerar point = [0,0,0], depois pensar na translacao
    # chegar no artigo do rodrigues, ou algo assim, para referenciar
    #? chamar rotation ?
    def rotate(self, axis, angle, axispoint=0):
        """
        Rotate the block around a given axis by a given angle.

        Parameters
        ----------
        axis : ndarray or list or tuple, shape (3,)
            Vector axis of rotation.
        angle : float
            Angle of rotation [rad].
        axispoint : ndarray or list or tuple, shape (3,), optional
            Point on the axis of rotation, by default (0,0,0).

        Notes
        -----
        Implements the Rodrigues' rotation formula [1].
        The axis vector can be of any length.

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

        """
        k = np.asarray(axis)/np.linalg.norm(axis)
        K = np.array([[0,-k[2],k[1]],
                      [k[2],0,-k[0]],
                      [-k[1],k[0],0]])
        R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
        # R = scipy.linalg.expm(angle*K)
        self._T = R @ self._T

        return R

    #? chamar reflection, plane_symmetry ?
    def mirror(self, normal, planepoint=0):
        """
        Mirror the block with respect to a given plane.

        Parameters
        ----------
        normal : ndarray or list or tuple, shape (3,)
            Normal vector of the mirror plane.
        planepoint : ndarray or list or tuple, shape (3,), optional
            Point on the plane of mirroring, by default (0,0,0).

        Notes
        -----
        Implements the Householder transformation [1].
        The normal vector can be of any length.

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Householder_transformation

        """
        n = np.asarray(normal)/np.linalg.norm(normal)
        N = np.eye(3) - 2*np.outer(n,n)
        self._T = N @ self._T

        return N

    def invert_magnetization(self):
        #? self._T = -self.T ?
        pass

    def translate(self, vec):
        pass



class Parallelepiped(Block):

    def __init__(self,dimensions,center=None,magnetization=None):
        """
        A uniformly magnetized parallelepiped.

        Args:
        center:
            The center of the parallelepiped [m]; (xc, yc, zc).
        dimensions:
            The widths of the parallelepiped [m]; (wx, wy, wz).
        magnetization:
            The magnetization vector of the parallelepiped [T]; (mx, my, mz).
        """
        super().__init__(magnetization)

        self._pc = np.asarray(center) if center else np.zeros(3)
        self._w = np.asarray(dimensions)

    @property
    def center(self):
        """Center of the rectangular block [m]; (xc, yc, zc)"""
        return self._pc
    
    @center.setter
    def center(self,center):
        self._pc = np.asarray(center)

    @property
    def width(self):
        """Widths of the rectangular block [m]; (wx, wy, wz)"""
        return self._w

    def faces(self):

        square = [[1,1],[-1,1],[-1,-1],[1,-1]]
        cube   = []

        for f in range(3):
            face1 = np.insert(square,f, 1,axis=1)
            face2 = np.insert(square,f,-1,axis=1)
            cube.extend([face1,face2])

        return self.center + (cube*self.width/2)
    
    def vertices(self):
        return np.unique(self.faces().reshape(-1,3),axis=0)   

    
    #todo: permitir usar r0 como ndarray de pontos
    def Q(self,r0):
        
        r0 = np.asarray(r0)
        pc = self.center
        w = self.width

        I, J, K = np.meshgrid([1,2],[1,2],[1,2])
        I, J, K = I.reshape(-1), J.reshape(-1), K.reshape(-1)

        X, Y, Z = [r0[i] - pc[i] + ((-1)**IDX)*w[i]/2
                       for i,IDX in enumerate([I,J,K])]
        
        arrQ = np.empty([3,3])
        
        for m in range(3):
            for n in range(3):
                
                P1, P2, P3 = np.roll([X,Y,Z], m+n, axis=0)

                # validate P1
                mask = P1 != 0
                P1, P2, P3 = P1[mask], P2[mask], P3[mask]
                Im, Jm, Km = I[mask], J[mask], K[mask]

                if m==n:
                    q = np.arctan( P2*P3/( P1*np.sqrt(P1**2+P2**2+P3**2) ) )
                    arrQ[m,n] = np.sum( ((-1)**(Im+Jm+Km+1)) * q )
                elif m<n:
                    q = np.log( P1 + np.sqrt(P1**2+P2**2+P3**2) )
                    arrQ[m,n] = np.sum( ((-1)**(Im+Jm+Km)) * q )
                else:
                    arrQ[m,n] = arrQ[n,m]

        return (1/(4*np.pi)) * arrQ
    

    def isPointInsideBlock(self, r0):
        r0 = np.asarray(r0)
        return np.all(np.abs(r0-self.center) < self.width/2)
    


class Polyhedron(Block):

    # por enquanto a ordem dos vertices no faces_indices deve ser anti-horaria
    # para calcular vetor normal para fora do poliedro corretamente.
    #todo: incluir ordenamento dos indices das faces para tornar anti-horaria,
    #todo: nao e' necessario a principio ter a ordem, ela pode ser encontrada
    def __init__(self,vertices,faces_indices,magnetization=None):
        """A uniformly magnetized polyhedron.

        Args:
            magnetization (ndarray, shape (3,)): The magnetization vector of
            the polyhedron, in Tesla.
        """
        super().__init__(magnetization)

        self.vertices = vertices
        self.faces_indices = faces_indices


    def vertex(self,sigma,p):
        Nsig = len(self.faces_indices[sigma])
        idx = self.faces_indices[sigma][p % Nsig] # p (mod Nsig)
        return np.asarray(self.vertices[idx])

    def face(self,sigma):
        Nsig = len(self.faces_indices[sigma])
        return [self.vertex(sigma,p) for p in range(Nsig)]

    def normal(self,sigma):
        
        facesig = self.face(sigma)
        v1 = facesig[1] - facesig[0]
        v2 = facesig[2] - facesig[0]

        nvec = np.cross(v1,v2)
        n = nvec/np.linalg.norm(nvec)

        return n


    def Q(self,r0):

        # rotation matrix, from z to n
        def T(sigma):
            nx, ny, nz = self.normal(sigma)
            arT = np.array([[ (ny**2)/(1+nz) + nz ,    -nx*ny/(1+nz)    , nx],
                            [   -nx*ny/(1+nz)     , (nx**2)/(1+nz) + nz , ny],
                            [       -nx           ,         -ny         , nz]])
            return arT

        # rotated point
        def r(sigma,p,r0):
            r0 = np.asarray(r0)
            v = self.vertex(sigma,p)
            Tinv = np.linalg.inv(T(sigma))
            return Tinv @ (v - r0)
        
        # element of Q for every face vertex
        def q(x1,x2,a,b,z):

            def rho(x,a,b,z):
                return np.sqrt(x**2 + (a*x+b)**2 + z**2)
            
            def f1(x,a,b,z):
                return a*x + b + rho(x,a,b,z)

            def f2(x,a,b,z):
                return a*b + (1+a**2)*x + np.sqrt(1+a**2)*rho(x,a,b,z)
            
            def f3(x,a,b,z):
                return 2*(x**2+z**2)*a*b*(z**2) + (a*(z**2)+b*x) * ((a**2)*(z**2)+b**2) * f1(x,a,b,z)
            
            def f4(x,a,b,z):
                return ( ((a**2)*(z**2)-b**2) * (x**2+z**2) + (a*x-b) * ((a**2)*(z**2)+b**2) * f1(x,a,b,z) ) * z
            
            d = ((1+a**2)*(z**2)+b**2) * (4*((a**2)*(z**2)+b**2)*(a**2)*(b**2) - ((a**2)*(z**2)-b**2)**2)

            def theta(x):
                return np.heaviside(x,1)

            def beta(x,a,b,z):
                return theta( (b-a*x) * ( ((a**2)*(z**2)-b**2) * (x**2+z**2) + ((a**2)*(x**2)-b**2) * ((a**2)*(z**2)+b**2) ) )

            def csi(k,a,b,z):
                num = (((a**2)*(z**2)+b**2)**2)*a*b + ((-1)**(k+1)) * np.abs((a**2)*(z**2)-b**2) * np.sqrt(d)
                den = 4*(a**2)*(b**4) - (1+a**2) * (((a**2)*(z**2)-b**2)**2)
                return num/den

            def C(x1,x2,a,b,z):
                soma = 0
                for k in range(2):
                    soma += ((-1)**k) * beta(csi(k,a,b,z),a,b,z) * theta((x1-csi(k,a,b,z))*(csi(k,a,b,z)-x2)) * np.sign((b-a*csi(k,a,b,z))*f3(csi(k,a,b,z),a,b,z))  
                return np.pi * np.sign((x2-x1)*z) * soma
            

            def qz(x1,x2,a,b,z):
                # c = np.where(d>0,C(x1,x2,a,b,z),0) # problem: computes C() every time
                c = C(x1,x2,a,b,z) if d>0 else 0
                return np.arctan(f3(x1,a,b,z)/f4(x1,a,b,z)) - np.arctan(f3(x2,a,b,z)/f4(x2,a,b,z)) + c
            
            def qy(x1,x2,a,b,z):
                # print('f2 num:',f2(x1,a,b,z))
                # print('f2 den:',f2(x2,a,b,z))
                return ((1+a**2)**(-1/2)) * np.log(f2(x1,a,b,z)/f2(x2,a,b,z)) #

            def qx(x1,x2,a,b,z):
                return -a*qy(x1,x2,a,b,z) + np.log(f1(x1,a,b,z)/f1(x2,a,b,z)) #

            return np.array([qx(x1,x2,a,b,z), qy(x1,x2,a,b,z), qz(x1,x2,a,b,z)])

        # integral for each face
        def Qsig(sigma,r0):
            Nsig = len(self.faces_indices[sigma])

            vec = 0

            for p in range(Nsig):
                
                x , y , z = r(sigma, p ,r0)
                xx, yy, _ = r(sigma,p+1,r0)

                # print(sigma,p,x,y,z)
                # print(sigma,p,xx,yy,z)

                if xx != x:

                    x1 = x
                    x2 = xx
                    a = (yy-y)/(xx-x)
                    b = (xx*y-x*yy)/(xx-x)

                    # print(sigma,p,'x1:',x1,'x2:',x2,'a:',a,'b:',b)
                    
                    vec += q(x1,x2,a,b,z)
            
            return vec

        arrQ = 0
        for sigma in range(len(self.faces_indices)):
            arrQ += np.outer( T(sigma) @ Qsig(sigma,r0) , self.normal(sigma) )

        return (1/(4*np.pi)) * arrQ


    def isPointInsideBlock(self, r0):
        r0 = np.asarray(r0)

        N = len(self.faces_indices)

        p_on_faces = np.array([self.vertex(sigma,0) for sigma in range(N)])
        n_of_faces = np.array([self.normal(sigma) for sigma in range(N)])

        # dot prod between outward normal of faces and vector from face to r0
        dot = np.sum(n_of_faces*(r0-p_on_faces),axis=1) 

        return np.all( dot < 0 )



class Magnet:

    def __init__(self,blocks: list[Block]):
        self.blocks = blocks


    def Mfield(self,r0):
        fld = [mblock.Mfield(r0) for mblock in self.blocks]
        return np.sum(fld,axis=0)
    
    def Hfield(self,r0):
        fld = [mblock.Hfield(r0) for mblock in self.blocks]
        return np.sum(fld,axis=0)

    def Bfield(self,r0):
        fld = [mblock.Bfield(r0) for mblock in self.blocks]
        return np.sum(fld,axis=0)
    
