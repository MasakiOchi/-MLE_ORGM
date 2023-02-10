import numpy as np
from scipy.sparse import csr_matrix, triu
import MLE_tools as ga
import graph_tool.all as gt

def MLE_ORGM(g, K, beta, eta, sample_prob, n_sm):
    
    Adj_csr = gt.adjacency(g)
    N = g.num_vertices()
    Adj_u =triu(Adj_csr,k=1,format="csr")
    M = Adj_u.nnz
    vertex_sequence_init = ga.spectral_ordering(g) # initialize vertex_sequence by spectral ordering
    
    l_max = -1000000
    vertex_sequence_max = np.zeros(N)
    a_max = np.zeros(K)
    pin_max = 0
    pout_max = 0
    a_init = np.random.uniform(1, N/K/2, (n_sm, K)) # initial values of {a_k}
    
    for sm in range(n_sm):
        
        vertex_sequence = vertex_sequence_init.copy()
        a = a_init[sm]
        
        omega_in_size = ga.Omega_in_size(a, N)
        if omega_in_size == 0:
            continue
        omega_in_population = ga.Omega_in_population(vertex_sequence, Adj_u, a, N)
        
        # random pairs of vertices to be flipped in greedy_flip
        greedy_s=np.random.permutation(N) 
        greedy_t=np.random.permutation(N)
        ind = 0
        
        num_itr=0
        l = -10000000
        
        while num_itr<100:
            l0 = l
            # MLE of p_in and p_out
            pin, pout = ga.MLE_p(N, M, omega_in_size, omega_in_population)

            if pout >= pin:
                break
                
            # MLE of {a_k}
            a = ga.a_GD(a, Adj_u, vertex_sequence, pin, pout, N, M, beta, eta, 100) # update {a}
            
            omega_in_size = ga.Omega_in_size(a, N)
            if omega_in_size == 0:
                l = -10000000
                break
                
            # Greedy update of the vertex_sequence
            for i in range(int(N*sample_prob)):
                vertex_sequence = ga.greedy_flip(Adj_csr, a, vertex_sequence, pin, pout, N, greedy_s[ind], greedy_t[ind])
                if ind+1 == N:
                    greedy_s = np.random.permutation(N)
                    greedy_t = np.random.permutation(N)
                    ind=0
                else:
                    ind+=1
                    
            omega_in_population = ga.Omega_in_population(vertex_sequence, Adj_u, a, N)
            l = ga.log_likelihood(N, M, pin, pout, omega_in_size, omega_in_population)
            if abs(l-l0) < 0.000001:
                break
            num_itr+=1
            
        if l_max < l:
            l_max = l
            a_max = a
            pin_max, pout_max = pin, pout
            vertex_sequence_max = vertex_sequence
    return vertex_sequence_max, a_max, pin_max, pout_max, l_max