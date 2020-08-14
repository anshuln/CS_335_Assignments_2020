import numpy as np
np.set_printoptions(precision=16)
import math

#algorithm to find the index of first non-zero element in a column from position start_row 
def find_nonzero_element_index(col, start_row):
    #print(col)
    ret_index = -1
    for k in np.arange(start_row,col.size):
        #print('is close to zero: %s' %(math.isclose(np.abs(col[k]), 0., abs_tol=1e-13)))
        if not math.isclose(np.abs(col[k]), 0., abs_tol=1e-13):
            ret_index = k
            break
    #print ('ret index: %d' %(ret_index))
    return ret_index

#algorithm to convert a matrix A to rref
def find_rref(A):
    #print(A.shape)
    m = (A.shape)[0] #num rows
    n = (A.shape)[1] #num cols
    print('num rows: %d' %(m))
    print('num cols: %d' %(n))

    j = 0
    r = 0
    for j in np.arange(0,n):
        non_zero_row_index = find_nonzero_element_index(A[:,j],r)
        #print('r: %d' %(r))
        if non_zero_row_index != -1: 
            #write the code to make the pivot element in the non-zero row as 1 
            #and to make all other elements in the pivot column as zero  
            
            #you can check your code by printing the intermediate matrix using the following statement
            #print('rref process in col: %d ' %(j))
            #print(A)
            r = r+1
    return (A)

#returns the non-zero row indices and dimension of rowspace(A)
def find_rowspace_basis_dim(rref_A):
    m = (rref_A.shape)[0]
    non_zero_indices = [] 
    start_index = 0
    for i in np.arange(0,m):
        non_zero_pos = find_nonzero_element_index(rref_A[i].T,start_index)
        if non_zero_pos != -1:
            start_index = non_zero_pos
            non_zero_indices.append(non_zero_pos)
        else:
            break
    return non_zero_indices, len(non_zero_indices)


if __name__ == '__main__':
    Matrices = [] 
    Mnames = [] 
    
    A = np.matrix([[1.,2.,0.],[4.,5.,0.],[0.,0.,0.]])
    Matrices.append(A)
    Mnames.append('A')
    
    B = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    Matrices.append(B)
    Mnames.append('B')
    
    C = np.matrix([[1.,2.,3.,4.,5.],[2.,3.,4.,5.,6.]])
    Matrices.append(C)
    Mnames.append('C')
    
    D = np.matrix([[1.,2.,3.], [2.,4.,6], [7.,8.,9], [3.,6.,9], [10.,14.,18.] ] )
    Matrices.append(D)
    Mnames.append('D')
    
    E = np.matrix([[ 0., -1., -2.],[ 1.,  0., -1.],[ 2.,  1.,  0.]])
    Matrices.append(E)
    Mnames.append('E')
    
    F = np.matrix([[  1., 1/2., 1/3., 1/4., 1/5.],[  2.,   1., 2/3., 1/2., 2/5.],[  3., 3/2.,   1., 3/4., 3/5.],[  4.,   2., 4/3.,   1., 4/5.],[  5., 5/2., 5/3., 5/4.,   1.]])
    Matrices.append(F)
    Mnames.append('F')
    
    G = np.matrix([[0., 1., 0., 0., 1., 1., 0., 0., 0., 0.],[1., 0., 1., 0., 0., 0., 1., 0., 0., 0.],[0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],[0., 0., 1., 0., 1., 0., 0., 0., 1., 0.],[1., 0., 0., 1., 0., 0., 0., 0., 0., 1.],[1., 0., 0., 0., 0., 0., 0., 1., 1., 0.],[0., 1., 0., 0., 0., 0., 0., 0., 1., 1.],[0., 0., 1., 0., 0., 1., 0., 0., 0., 1.],[0., 0., 0., 1., 0., 1., 1., 0., 0., 0.],[0., 0., 0., 0., 1., 0., 1., 1., 0., 0.]])
    Matrices.append(G)
    Mnames.append('G')
    
    H = np.matrix([[1., 2., 3., 4., 8.],[8., 1., 2., 3., 4.],[4., 8., 1., 2., 3.],[3., 4., 8., 1., 2.],[2., 3., 4., 8., 1.]])
    Matrices.append(H)
    Mnames.append('H')
    
    I = np.matrix([[ 0.,  3.,  6.,  9., 12., 15., 18., 21.],[ 1.,  4.,  7., 10., 13., 16., 19., 22.],[ 2.,  5.,  8., 11., 14., 17., 20., 23.]])
    Matrices.append(I)
    Mnames.append('I')
    
    J = np.matrix([[0., 0., 0., 0., 2.],[1., 0., 0., 0., 3.],[0., 1., 0., 0., 4.],[0., 0., 1., 0., 5.],[0., 0., 0., 1., 6.]])
    Matrices.append(J)
    Mnames.append('J')
    
    K = np.matrix([[6., 5., 4., 3., 2.], [1., 0., 0., 0., 0.],[0., 1., 0., 0., 0.],[0., 0., 1., 0., 0.],[0., 0., 0., 1., 0.]])
    Matrices.append(K)
    Mnames.append('K')
    
    L = np.matrix([[0., 1., 0., 0., 0.],[0., 0., 1., 0., 0.],[0., 0., 0., 1., 0.],[0., 0., 0., 0., 1.],[2., 3., 4., 5., 6.]])
    Matrices.append(L)
    Mnames.append('L')
    
    M = np.matrix([[6., 1., 0., 0., 0.],[5., 0., 1., 0., 0.],[4., 0., 0., 1., 0.],[3., 0., 0., 0., 1.],[2., 0., 0., 0., 0.]])
    Matrices.append(M)
    Mnames.append('M')
    
    N = np.matrix([[  90.,  -80.,   56., -448., -588.],[  60.,    0.,   28., -324., -204.],[  60.,  -72.,   32., -264., -432.],[  30.,  -16.,   16., -152., -156.],[ -10.,   -8.,   -4.,   60.,    8.]])
    Matrices.append(N)
    Mnames.append('N')
    
    O = np.matrix([[ -15.,   -4.,   83.,   35.,  -24.,   47.,  -74.,  50.],[ -16.,   -7.,   94.,   34.,  -25.,   38.,  -75.,   50.,],[  89.,   34., -513., -196.,  141., -235.,  426., -285.],[  17.,    6.,  -97.,  -38.,   27.,  -47.,   82.,  -55.],[   7.,    3.,  -41.,  -15.,   11.,  -17.,   33.,  -22.],[  -5.,   -2.,   29.,   11.,   -8.,   13.,  -24.,   16.]])
    Matrices.append(O)
    Mnames.append('O')
    
    P = np.matrix([[    1.,     2.,     8.,   -35.,  -178.,  -239.,  -284.,   778.],[    4.,     9.,    37.,  -163.,  -827., -1111., -1324.,  3624.],[    5.,     6.,    21.,   -88.,  -454.,  -607.,  -708.,  1951.],[   -4.,    -5.,   -22.,    97.,   491.,   656.,   779., -2140.],[    4.,     4.,    13.,   -55.,  -283.,  -377.,  -436.,  1206.],[    4.,    11.,    43.,  -194.,  -982., -1319., -1576.,  4310.],[   -1.,    -2.,   -13.,    59.,   294.,   394.,   481., -1312.]])
    Matrices.append(P)
    Mnames.append('P')
    
    Q = np.matrix ([[    1.,     1.,     7.,   -29.,   139.,   206.,   413.],[   -2.,    -1.,   -10.,    41.,  -197.,  -292.,  -584.],[    2.,     5.,    27.,  -113.,   541.,   803.,  1618.],[    4.,     0.,    14.,   -55.,   268.,   399.,   798.],[    3.,     1.,     8.,   -32.,   152.,   218.,   412.],[   -3.,    -2.,   -18.,    70.,  -343.,  -506., -1001.],[    1.,    -2.,    -1.,     1.,    -2.,     9.,    52.]])
    Matrices.append(Q)
    Mnames.append('Q')
    
    for m in np.arange(0,len(Mnames)):
        print('matrix %s:' %(Mnames[m]))
        print(Matrices[m])
        #find the rref 
        rref_m = find_rref(Matrices[m])
        print('rref of matrix %s:' %(Mnames[m]))
        print(rref_m)
        
        #find the rowspace and its dimension 
        rowspace_basis_indices, rowspace_dim = find_rowspace_basis_dim(rref_m)
        print(rowspace_basis_indices)
        print('Rowspace Basis of matrix %s: ' %(Mnames[m]))
        print( (Matrices[m])[rowspace_basis_indices])
        print('Rowspace dimension of matrix %s: %d' %(Mnames[m],rowspace_dim))
        print('***********************************************')