from itertools import combinations
import numpy as np
import scipy.special
from scipy import sparse as sp
from numpy import linalg as LA
import scipy.linalg as la
from scipy.optimize import brentq


def Dim(N_particle, M_site):
    """Computes dimension of the Hilbert space"""
    return int(scipy.special.binom(N_particle + M_site - 1, N_particle))

def occuNum(N, M):
    """
    Generating all possible bases vector in number bases
    with M sites and N particles
    """
    for c in combinations(range(N + M - 1), M - 1):
        yield [b - a - 1 for a, b in zip((-1,) + c, c + (N + M - 1,))]

def basisArray(N, M):
    """
    Generates and save all basis states in one array D X M
    ordenate lexicographics
    """
    D = Dim(N, M)
    generator = occuNum(N, M)
    Base = sorted([list(next(generator)) for i in range(D)], reverse=True)
    return Base

def crea(site: int, vector, N):
    """
    Apply the creation operator at place "site" in state "vector",
    returns a list where the first element is the resultant state
    not normalized and the second element is the normalization factor
    """
    if (type(vector)==list and vector[site] < N):
      vector_f = np.copy(vector)
      vector_f[site] = vector[site] + 1
      return vector_f, np.sqrt(vector[site]+1)
    else:
      return 0, 0

def ani(site: int, vector):
    """
    Apply the annihilation operator at place "site" in state "vector",
    returns a list where the first element is the resultant state
    not normalized and the second element is the normalization factor
    """
    if (type(vector)==list and vector[site] > 0):
      vector_f = vector.copy()
      vector_f[site] = vector[site] - 1
      return vector_f, np.sqrt(vector[site])
    else:
        return 0, 0

def OperatorCreaAni(v,i,j,N):
    """
    This function applies the operator b ^ \ dag_i b_j and
    returns the resulting vector without normalizing
    """
    bb_dag = crea(i,ani(j,v)[0],N)[0]
    if type(bb_dag) == int:
      return 0, 0
    else:
      return bb_dag

def Tag_base(Base,vector):
    Dictionary_base = {tuple(Base[i]): i for i in range(len(Base))}
    index_vector = Dictionary_base.get(vector)
    return index_vector

def HamiltonianoSparse(Base,J,U,N,M):
    Dictionary_base = {tuple(Base[i]): i for i in range(len(Base))}
    columna, renglon, datos = [], [], []
    for i in range(len(Base)):
        for m in range(M-1):
                col=Dictionary_base.get(tuple(OperatorCreaAni(Base[i],m,m+1,N)))

                if col!=None:
                    columna.append(col)
                    renglon.append(i)
                    dat=-J*np.sqrt(Base[i][m+1]*(Base[i][m]+1))

                    datos.append(dat)

                col2=Dictionary_base.get(tuple(OperatorCreaAni(Base[i],m+1,m,N)))

                if col2!=None:

                    columna.append(col2)
                    renglon.append(i)
                    dat=-J*np.sqrt(Base[i][m]*(Base[i][m+1]+1))

                    datos.append(dat)

    renglon = renglon + list(range(len(Base)))
    columna = columna + list(range(len(Base)))
    diag = [0.5*U*sum(np.array(row)*np.array(row) - np.array(row)) for row in Base]
    datos = datos + diag

    # row indices
    row_ind = np.array(renglon)

    # column indices
    col_ind = np.array(columna)
    # data to be stored in COO sparse matrix
    data = np.array(datos, dtype=float)
    # create COO sparse matrix from three arrays
    mat_coo = sp.coo_matrix((data, (row_ind, col_ind)))

    H=mat_coo.toarray()

    E, V = la.eigh(H)

    return E, V

def initial_state(index_state,Dimension):
    psi0=np.zeros(Dimension)
    for i in index_state:
        psi0[i]=1
    return psi0
###############################################################################
#       CONSTRUYE EL VECTOR DE ESTADO EN FUNCION DEL TIEMPO
###############################################################################
def psit(t, val,vec,psi0):
    psitin=np.zeros(len(val),dtype=complex)
    for i in range(len(val)):
        psitin+=np.exp(-1j*val[i]*t)*np.dot(vec[0:len(vec),i],psi0)*vec[0:len(vec),i]
    return psitin


def Psi_Matrix(psi):
    psi2=psi.conjugate()
    psiT= np.reshape(psi2, (len(psi),1))
    psitemp=np.kron(psiT,psi)
    return psitemp

def dimension(N,M):
    if M == 1:
        return int(N+1)
    else:
        DT=0
        for i in range(1,N+1):
            DT+=Dim(i,M)

        return int(DT+1)

def Base_sub_system(N_part,M_site):
    Base_sub =[]
    for i in range(1,N_part+1):
        Base_sub += basisArray(i, M_site)
    return sorted(Base_sub,reverse=True)

def sub_system(Base, SubSystem):
    return [Base[site] for site in SubSystem]

def complement_sub_system(Base, SubSystem, M):
    return [Base[site] for site in range(M) if site not in SubSystem]

def trace_elements(B,Bn2,SubSystem,M):
    def Match_sub_sist(Bn2,B,SubSystem):
          """Find all elements of the complete base compatible with the subsytem base"""
          ConjEtiq=[]
          ConjEtiqTemp=[]

          for l in range(len(Bn2)):
              for k in range(len(B)):

                  if Bn2[l]== sub_system(B[k],SubSystem):
                      ConjEtiqTemp.append(k)

              ConjEtiq.append(ConjEtiqTemp)
              ConjEtiqTemp=[]

          return ConjEtiq

    conj = Match_sub_sist(Bn2,B,SubSystem)

    #print(conj)

    def trace_elements_nm(n,m):
          if sum(Bn2[n]) == sum(Bn2[m]):
              Tr_nm = []
              for Cn in conj[n]:
                  for Cm in conj[m]:
                      if complement_sub_system(B[Cn],SubSystem, M) == complement_sub_system(B[Cm],SubSystem, M):
                          Tr_nm.append([Cn,Cm])

              return Tr_nm
          else:
              return

    #print(trace_elements_nm(2,3))

    Tr_elements = []
    for n in range(len(Bn2)):
        for m in range(len(Bn2)):
            tr_nm = trace_elements_nm(n,m)
            if tr_nm != None:
                Tr_elements.append([n,m,tr_nm])

    #print(round(time.time() - tim_exec,3), "elemtos tiempo")
    return Tr_elements

def matriz2a6sitios(ro, totalij, tamSub):
    """
    Esta funcion toma la lista totalij dada por la funcion elementosij y construye
    la matriz de densidad reducida
    """
    ro2sitios=np.zeros((tamSub,tamSub),dtype=complex)
    for l in totalij:
        for k in l[2]:
            ro2sitios[l[0]][l[1]]+=ro[k[0]][k[1]]

    #Trazaro=-1*np.log(np.trace(np.dot(ro2sitios,ro2sitios)).real)
    return ro2sitios


def Partial_trace(Base, Base_subsystem, Subsystem, elements_trace, initial_state_psi0, Eigenvalue, Eigenstate, Tiempo):
    state = psit(Tiempo,Eigenvalue,Eigenstate,initial_state_psi0)
    Matrix_state = Psi_Matrix(state)
    ro_sub = matriz2a6sitios(Matrix_state,elements_trace,len(Base_subsystem))
    entropy =-np.log(np.trace(np.dot(ro_sub,ro_sub)))

    return entropy


def EnergiaPromedio(beta,EnT, initial_energy):
        """calcula la energia promedio en el ensemble canonico"""

        suma=0
        sumae=0
        for n in range(len(EnT)):
            #print(-beta*EnT[n], beta, suma)
            suma+=np.exp(-beta*EnT[n])

        for n in range(len(EnT)):
            sumae+=np.exp(-beta*EnT[n])*EnT[n]

        return sumae/suma.real- initial_energy





def Canonical_Matrix(EnT,beta):
     """
     Esta funcion calcula la matriz de densidad en el ensamble canonico
     para una temperatura T=1/beta
     """

     suma=0
     for n in EnT:
         suma+=np.exp(-beta*n)
     #print suma
     Ro=np.zeros((len(EnT),len(EnT)))

     for l in range(len(EnT)):
         Ro[l][l]=np.exp(-beta*EnT[l])/suma


     #print(time()-tim,suma)
     return Ro

def Funt(J,EnT,prome):
    """Calcula la temperatura en el ensemble canonico"""


    if J<3:
        sol1 = brentq(lambda Beta: EnergiaPromedio(Beta, EnT, prome), -10,20, maxiter=100)

    if J>3:
        sol1 = brentq(lambda Beta: EnergiaPromedio(Beta, EnT, prome),-10,10,maxiter=100)


    return sol1


def Change_base(Matriz, Eigenstates):
    Inverse = LA.inv(Eigenstates)
    Matrix_new_base = Eigenstates @ Matriz @ Inverse
    return  Matrix_new_base


def Energy(state,U):
    state = np.array(state)
    return 0.5*U*sum(state*state - state)


def EnergyState(U,args):
    print(len(args),"args")
    if len(args)>1:
        Ener = 0
        for i in args:
            state = np.array(i)
            print (i)
            Ener =Ener + 0.5*U*sum(state*state - state)    
        return Ener
    if len(args)==1:
        state = np.array(args[0])
        return 0.5*U*sum(state*state - state)
        
