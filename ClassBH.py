#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:47:09 2021
@author: Alejandro
"""
from sys import getsizeof
from time import time as time
import scipy.special
from itertools import combinations
import numpy as np
from scipy import sparse as sp
from numpy import linalg as LA
import scipy.linalg as la


class Base1D:
    def __init__(self, N, M):
        self.M = M
        self.N = N

    def Dimension(self, N_site, M_site):
        """Computes dimension of the Hilbert space"""
        return int(scipy.special.binom(N_site + M_site - 1, N_site))

    def GeneratorBase(self):
        """
        Generating all possible bases vector in number bases
        with M sites and N particles
        """

        for c in combinations(range(self.N + self.M - 1), self.M - 1):
            yield [b - a - 1 for a, b in zip((-1,) + c,
                   c + (self.N + self.M - 1,))]

    def basisArrays(self, Ns, Ms):
        """
        Generates and save all basis states in one array D X M
        ordenate lexicographics
        """
        D = self.Dimension(Ns, Ms)
        generator = self.GeneratorBase()
        self.Base_list = sorted([list(next(generator)) for i in range(D)],
                                reverse=True)
        return self.Base_list


class Operators:

    def Crea(self, site, vector, N):
        """
        Apply the creation operator at place "site" in state "vector",
        returns a list where the first element is the resultant state
        not normalized and the second element is the normalization factor
        """
        if (type(vector) == list and vector[site] < N):
            vector_f = np.copy(vector)
            vector_f[site] = vector[site] + 1
            return vector_f, np.sqrt(vector[site]+1)
        else:
            return 0, 0

    def Ani(self, site: int, vector):
        """
        Apply the annihilation operator at place "site" in state "vector",
        returns a list where the first element is the resultant state
        not normalized and the second element is the normalization factor
        """
        if (type(vector) == list and vector[site] > 0):
            vector_f = vector.copy()
            vector_f[site] = vector[site] - 1
            return vector_f, np.sqrt(vector[site])
        else:
            return 0, 0

    def Hopp(self, site_initial, site_final, vector, N):
        """
        This function applies the hopp operator (b ^ \ dag_i b_j) and
        returns the resulting vector without normalizing
        """
        vector_hopp = self.Crea(site_final, self.Ani(site_initial,
                                                     vector)[0], N)[0]
        if type(vector_hopp) == int:
            return 0, 0
        else:
            return vector_hopp


class Hamiltonian(Base1D, Operators):

    def __init__(self, J, U, N, M):
        Base1D.__init__(self, N, M)
        Operators.__init__(self)
        self.J = J
        self.U = U
        self.D = self.Dimension(self.N, self.M)
        self.Base = self.basisArrays(self.N, self.M)


    def BuildHamiltonian(self):

        Dictionary_base = {tuple(self.Base[i]): i for i in range(self.D)}
        columna, renglon, data = [], [], []

        for i in range(self.D):
            for m in range(self.M-1):
                    hopp_vec = tuple(self.Hopp(m, m+1, self.Base[i], self.N))
                    hopp_vec_index = Dictionary_base.get(hopp_vec)
                    if hopp_vec_index:
                        columna.append(hopp_vec_index)
                        renglon.append(i)
                        dat = - self.J*np.sqrt(self.Base[i][m]*(self.Base[i][m+1]+1))
                        data.append(dat)
                    hopp_vec_c = tuple(self.Hopp(m+1, m, self.Base[i], self.N))
                    hopp_vec_c_index = Dictionary_base.get(hopp_vec_c)
                    if hopp_vec_c_index:
                        columna.append(hopp_vec_c_index)
                        renglon.append(i)
                        dat = -self.J*np.sqrt(self.Base[i][m+1]*(self.Base[i][m]+1))
                        data.append(dat)
        # Built diagonal elements
        renglon = renglon + list(range(self.D))
        columna = columna + list(range(self.D))
        diag = [0.5*self.U*sum(np.array(row)*np.array(row) - np.array(row))
                for row in self.Base]
        data = data + diag
        # store row indices
        row_ind = np.array(renglon)
        # store column indices
        col_ind = np.array(columna)
        # data to be stored in COO sparse matrix
        data = np.array(data, dtype=float)
        # create COO sparse matrix from three arrays
        mat_coo = sp.coo_matrix((data, (row_ind, col_ind)))
        H = mat_coo.toarray()
        self.E, self.V = la.eigh(H)
        return self.E, self.V

    def Initial_state(self, index_state):
        self.psi0 = np.zeros(462)
        for i in index_state:
            self.psi0[i] = 1
        return self.psi0

    def Psit(self, t):
        psi_t = np.zeros(self.D, dtype=complex)
        for i in range(self.D):
            psi_t += np.exp(-1j*self.E[i]*t)*np.dot(self.V[:][i], self.psi0)*self.V[:][i]
        return psi_t

    def PsiMatrix(self, psi):
        psi2 = psi.conjugate()
        psiT = np.reshape(psi2, (len(psi), 1))
        psitemp = np.kron(psiT, psi)
        return psitemp
