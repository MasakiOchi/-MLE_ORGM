import numpy as np
from scipy.sparse import linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import graph_tool.all as gt
import matplotlib.pyplot as plt

# envelope function b(x_{ij})
def envelope_function(a, N, i, j):
    b = 0
    for k in range(len(a)):
        b += np.sqrt(2)*a[k]*(np.sin(np.pi*(k+1)*(i+j)/2/(N-1)))**2
    return b

# The size of \Omega_{in}
# If the envelope function leaks out of the upper-triangle, return 0.
def Omega_in_size(a,N): 
    s = 0
    boundary_bool = True
    for d in range(N-2): # i+j = even
        k = 2*(d+1)
        b = envelope_function(a, N, k, 0)
        if (b<=k) & (b<=2*N-k) & (b>=0):
            s += int(b/2)
        else:
            boundary_bool = False
            break
    if boundary_bool == True:
        for d in range(N-1): # i+j = odd
            k = 2*d+1
            b = envelope_function(a,N,k,0)
            if (b<=k) & (b<=2*N-k) & (b>=0):
                s += int((b+1)/2)
            else:
                boundary_bool = False
                break
    if boundary_bool == True:
        return s
    else:
        return 0

# The number of non-zero elements in \Omega_{in}
# Adj_u : the upper triangle of an adjacency matrix
def Omega_in_population(vertex_sequence, Adj_u, a, N): 
    p = 0
    for i in range(N):
        for ptr in range(Adj_u.indptr[i], Adj_u.indptr[i+1]):
            j = Adj_u.indices[ptr]
            if envelope_function(a, N, vertex_sequence[i], vertex_sequence[j])>=abs(vertex_sequence[i]-vertex_sequence[j]):
                p += 1
    return p

# Log-likelihood function
def log_likelihood(N, M, pin, pout, omega_in_size, omega_in_population): #elements_in=in_envelope(a,N)
    if pout == 0:
        L = omega_in_population*(np.log(pin))
        L -= pin*omega_in_size
    else:
        L = omega_in_population*(np.log(pin)-np.log(pout))
        L -= (pin-pout)*omega_in_size
        L += np.log(pout)*M
        L -= pout*N*(N-1)/2
    return L

# Maximum-likehood estimation of pin and pout
def MLE_p(N, M, omega_in_size, omega_in_population):
    pin = omega_in_population/omega_in_size
    pout = (M-omega_in_population)/(N*(N-1)/2-omega_in_size)
    return pin, pout

# Sigmoid function
def sigmoid(x, beta):
    if x*beta<-100:
        return 0
    else:
        return 1/(1+np.exp(-beta*x))

# 1/cosh^2(x)
def coth2(x):
    if abs(x)<100:
        return 1/(np.cosh(x))**2
    else:
        return 0

# derivative of L_{\beta} in terms of a_k (Eq.21)
def dL_coeff(a, Adj_u, vertex_sequence, pin, pout, N, beta, delta):
    y = np.zeros(len(a))
    
    # First term
    for i in range(N):
        for ptr in range(Adj_u.indptr[i], Adj_u.indptr[i+1]):
            j = Adj_u.indices[ptr]
            if pout == 0:
                c1 = np.log(pin)*np.sqrt(2)*beta*coth2(beta*(envelope_function(a,N,vertex_sequence[i],vertex_sequence[j])-abs(vertex_sequence[i]-vertex_sequence[j]))/2)/4
            else:
                c1 = (np.log(pin)-np.log(pout))*np.sqrt(2)*beta*coth2(beta*(envelope_function(a,N,vertex_sequence[i],vertex_sequence[j])-abs(vertex_sequence[i]-vertex_sequence[j]))/2)/4
            for k in range(len(a)):
                y[k] += c1*((np.sin(np.pi*(k+1)*(vertex_sequence[i]+vertex_sequence[j])/2/(N-1)))**2)
                
    # Second term : sum over elements around the boundary as in Eq.23
    for d in range(N): # i+j = even
        b = int(envelope_function(a,N,2*d,0)/2)
        for l in range(max(b-int(delta),0), b+int(delta)):
            if d+1+l<N:
                i = d+1+l
                if d-1-l>0:
                    j = d-1-l 
                    c2 = -(pin-pout)*np.sqrt(2)*beta*coth2(beta*(envelope_function(a,N,i,j)-abs(i-j))/2)/4
                    for k in range(len(a)):
                        y[k] += c2*((np.sin(np.pi*(k+1)*(i+j)/2/(N-1)))**2)
    for d in range(N-1): # i+j = odd
        b = int((envelope_function(a,N,2*d+1,0)+1)/2)
        for l in range(max(b-int(delta)-1,0), b+int(delta)):
            if d+l+1<N:
                i = d+l+1
                if d-l>0:
                    j = d-l
                    c3 = -(pin-pout)*np.sqrt(2)*beta*coth2(beta*(envelope_function(a,N,i,j)-abs(i-j))/2)/4
                    for k in range(len(a)):
                        y[k] += c3*((np.sin(np.pi*(k+1)*(i+j)/2/(N-1)))**2)
    return y


# Maximum log-likelihood estimation of {a} with stochastic gradient descent
def a_GD(a, Adj_u, vertex_sequence, pin, pout, N, M, beta, eta, m_itr):
    itr = 0
    while itr < m_itr: 
        y = dL_coeff(a, Adj_u, vertex_sequence, pin, pout, N, beta, 2)
        if np.linalg.norm(y)<1/10:
            break
        else:
            a += y*eta/(itr+1)
            itr += 1
    return a

# The variation of log-likelihood function after flipping l and m
# NOTE! : Here, we use the symmetric adjacency matrix.
def delta_likelihood(Adj, a, vertex_sequence, pin, pout, N, l, m):
    dl = 0
    if pout == 0:
        pout = 2/N/(N-1) # prevent log(0)
    for ptr1 in range(Adj.indptr[l], Adj.indptr[l+1]):
        j = Adj.indices[ptr1]
        if j != m:
            if envelope_function(a, N, vertex_sequence[m], vertex_sequence[j]) >= abs(vertex_sequence[m]-vertex_sequence[j]):
                if envelope_function(a, N, vertex_sequence[l], vertex_sequence[j]) < abs(vertex_sequence[l]-vertex_sequence[j]):
                    dl += np.log(pin)-np.log(pout)                    
            else:
                if envelope_function(a, N, vertex_sequence[l], vertex_sequence[j]) >= abs(vertex_sequence[l]-vertex_sequence[j]):
                    dl += -(np.log(pin)-np.log(pout))
                    
    for ptr2 in range(Adj.indptr[m], Adj.indptr[m+1]):
        i = Adj.indices[ptr2]
        if i != l:
            if envelope_function(a, N, vertex_sequence[i], vertex_sequence[l]) >= abs(vertex_sequence[i]-vertex_sequence[l]):
                if envelope_function(a, N, vertex_sequence[i], vertex_sequence[m]) < abs(vertex_sequence[i]-vertex_sequence[m]):
                    dl += np.log(pin)-np.log(pout)                    
            else:
                if envelope_function(a, N, vertex_sequence[i], vertex_sequence[m]) >= abs(vertex_sequence[i]-vertex_sequence[m]):
                    dl += -(np.log(pin)-np.log(pout))
    return dl

# NOTE! : Here, we use the symmetric adjacency matrix.
def greedy_flip(Adj, a, vertex_sequence, pin, pout, N, l, m):
    dl = delta_likelihood(Adj, a, vertex_sequence, pin, pout, N, l, m)
    if dl>0:
        vertex_sequence_flipped = vertex_sequence.copy()
        vertex_sequence_flipped[l] = vertex_sequence[m]
        vertex_sequence_flipped[m] = vertex_sequence[l]
        vertex_sequence = vertex_sequence_flipped.copy()
    return vertex_sequence

def permutation_matrix(vertex_sequence, N):
    P = np.zeros((N,N))
    for i in range(N):
        P[i,int(vertex_sequence[i])]=1
    return P

def spectral_ordering(g):
    # Extract the giant connected component.
    g_ = g.copy()
    l = gt.label_largest_component(g_)
    g_gcc = gt.GraphView(g_, vfilt=l) 
    # Reorder the giant connected component using the spectral ordering.
    Lap = gt.laplacian(g_gcc, norm=True) 
    L = csr_matrix.toarray(Lap)
    w, v = eigsh(L, which="SA", k=2)
    degrees = g_gcc.degree_property_map("out").fa
    degree_correction = 1/np.sqrt(degrees)
    v2_ = v[:,1]
    v2 = degree_correction*v2_
    vertex_sequence = np.argsort(np.argsort(v2))
    # Unify isolated vertices.
    vertex_sequence_ = np.array([])
    ind_gcc = 0
    ind_nc = 0
    for i in g_.vertices():
        if i in g_gcc.vertices():
            vertex_sequence_ = np.append(vertex_sequence_, vertex_sequence[ind_gcc]+ind_nc)
            ind_gcc += 1
        else:
            vertex_sequence_ = np.append(vertex_sequence_, int(i))
            ind_nc += 1
    return vertex_sequence_

def save_graphml(g, vertex_sequence, a, pin, pout, filepath):
    vertex_sequence = g.new_vertex_property("int")
    for i in range(len(vertex_sequence)):
        vertex_sequence[i] = vertex_sequence[i]
    g.vertex_properties["Vertex sequence"] = vertex_sequence

    g.properties[("g", "EnvelopeFunction")] = g.new_graph_property("vector<double>")
    g.graph_properties["EnvelopeFunction"] = a

    g.properties[("g", "Probability")] = g.new_graph_property("vector<double>")
    g.graph_properties["Probability"] = np.array([pin, pout])

    g.save(filepath)

### heatmap with clustering envelope function ###

# Fill i≦x≦i+1, j≦y≦j+1
def mesh(ax,i,j,N):
    ax.axvspan(i+1/5, i+1-1/5, 1-(j+1/5)/N, 1-(j+1-1/5)/N, color = "black")
    ax.axvspan(j+1/5, j+1-1/5, 1-(i+1/5)/N, 1-(i+1-1/5)/N, color = "black")
    
# Draw envelope function
def b(ax,a,N):
    b_array = np.zeros(10*N+1)
    x_array_u = np.zeros(10*N+1)
    y_array_u = np.zeros(10*N+1)
    x_array_d = np.zeros(10*N+1)
    y_array_d = np.zeros(10*N+1)
    for i in range(10*N+1):
        for k in range(len(a)):
            b_array[i] += a[k]*(np.sin(np.pi*(k+1)*i/(N*10)))**2
        x_array_u[i] = i/10+b_array[i]/np.sqrt(2)
        y_array_u[i] = N-i/10+b_array[i]/np.sqrt(2)
        x_array_d[i] = i/10-b_array[i]/np.sqrt(2)
        y_array_d[i] = N-i/10-b_array[i]/np.sqrt(2)
    ax.plot(x_array_u,y_array_u, c='r', ls='--', lw=2)
    ax.plot(x_array_d,y_array_d, c='r', ls='--', lw=2)
    

def edgelist_reordered(g, vertex_sequence):
    Adj_csr = gt.adjacency(g)
    N = g.num_vertices()
    P = permutation_matrix(vertex_sequence,N)
    Adj = csr_matrix.toarray(Adj_csr)
    adj_a = np.transpose(P)@Adj@P
    edgelist = []
    for i in range(N-1):
        for j in range(i+1,N):
            if adj_a[i,j]==1:
                edgelist.append((i,j))      
    return edgelist

def plot_heatmap(ax, g, vertex_sequence, a):
    edge_array = edgelist_reordered(g, vertex_sequence)
    N = g.num_vertices()
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    for l in edge_array:
        mesh(ax,l[0],l[1],N)
    b(ax,a,N)
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    ax.set_aspect('equal')
