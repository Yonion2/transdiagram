import networkx as nx
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import odeint
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.sparse.linalg import eigsh as largest_eigsh
#抽取txt中的数据
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.integrate import solve_ivp

def read_txt_direct(data):
    g = nx.read_edgelist(data, create_using=nx.DiGraph())
    return g

def load_as_caida(filepath):
    G = nx.DiGraph()
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue  # 跳过注释行
            
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # 跳过不完整行
                
            src, dst, rel_type = parts
            
            # 添加带属性的边
            G.add_edge(src, dst)
    
    # 添加AS节点属性（可选）
    nx.set_node_attributes(G, {node: {"type": "AS"} for node in G.nodes()})
    return G

# 计算B0的最大特征根
def cal_B0_largest(G, p_bg, p_bb, r_bb, r_gb):
    A = nx.adjacency_matrix(G).T
    n, m =A.get_shape()
    B0 = vstack([hstack([(3*r_gb+ r_bb)*A - (3*p_bg+p_bb-2)*csr_matrix(np.eye(n)), (3*r_gb+ r_bb)*A + (p_bb/2)*csr_matrix(np.eye(n)), (3*r_gb+ r_bb)*A + (p_bb/2)*csr_matrix(np.eye(n))]), 
            hstack([(3*r_gb+ r_bb)*A + (p_bb/2)*csr_matrix(np.eye(n)),  (3*r_gb+ r_bb)*A - (3*p_bg+p_bb-2)*csr_matrix(np.eye(n)), (3*r_gb+ r_bb)*A + (p_bb/2)*csr_matrix(np.eye(n))]),
            hstack([(3*r_gb+ r_bb)*A + (p_bb/2)*csr_matrix(np.eye(n)), (3*r_gb+ r_bb)*A + (p_bb/2)*csr_matrix(np.eye(n)), (3*r_gb+ r_bb)*A - (3*p_bg+p_bb-2)*csr_matrix(np.eye(n))])])
    evals, x_vec = largest_eigsh(B0, 1, which='LM')
    return evals-2
    
def equations(x, t, p_bg, p_bb, r_bb, r_gb,A):
    n_nodes = x.shape[0]//6
    dx = np.zeros_like(x)
    thetam1 = x[3*n_nodes:4*n_nodes]@A #$\gamma^{i-1j}$
    thetam2 = x[4*n_nodes:5*n_nodes]@A#$\gamma^{i-2j}$
    thetam3 = x[5*n_nodes:6*n_nodes]@A#$\gamma^{i-3j}$
    dx[:n_nodes] = p_bg *(x[3*n_nodes: 4*n_nodes] + x[4*n_nodes: 5*n_nodes] +  x[5*n_nodes: 6*n_nodes]) - 2*0.1*x[:n_nodes] - 3*r_gb*(thetam1+ thetam2+ thetam3)*x[:n_nodes] + 0.1*(x[2*n_nodes:3*n_nodes] + x[n_nodes:2*n_nodes])
    dx[n_nodes:2*n_nodes] = p_bg * (x[3*n_nodes: 4*n_nodes] + x[4*n_nodes: 5*n_nodes] +  x[5*n_nodes: 6*n_nodes]) - 2*0.1*x[n_nodes:2*n_nodes] - 3*r_gb*(thetam1+ thetam2+ thetam3)*x[n_nodes:2*n_nodes] + 0.1*(x[2*n_nodes:3*n_nodes] + x[:n_nodes])
    dx[2*n_nodes:3*n_nodes] = p_bg * (x[3*n_nodes: 4*n_nodes] + x[4*n_nodes: 5*n_nodes] +  x[5*n_nodes: 6*n_nodes]) - 2*0.1*x[2*n_nodes:3*n_nodes] - 3*r_gb*(thetam1+ thetam2+ thetam3)*x[2*n_nodes:3*n_nodes] + 0.1*(x[:n_nodes] + x[n_nodes:2*n_nodes])
    dx[3*n_nodes:4*n_nodes] = -(3*p_bg+p_bb)*x[3*n_nodes: 4*n_nodes] - (r_bb*thetam2 +r_bb*thetam3)*x[3*n_nodes: 4*n_nodes] + r_gb*(thetam1+ thetam2+ thetam3)*(x[:n_nodes]+ x[n_nodes:2*n_nodes]+ x[2*n_nodes:3*n_nodes]) \
                                + (r_bb/2)*thetam1*(x[4*n_nodes: 5*n_nodes]+ x[5*n_nodes: 6*n_nodes]) + (r_bb/2)*thetam2*(x[5*n_nodes: 6*n_nodes]) + (r_bb/2)*thetam3*(x[4*n_nodes: 5*n_nodes])\
                                + (p_bb/2)*(x[4*n_nodes: 5*n_nodes]+x[5*n_nodes: 6*n_nodes]) 
    dx[4*n_nodes:5*n_nodes] = -(3*p_bg+p_bb)*x[4*n_nodes:5*n_nodes] - (r_bb*thetam1 +r_bb*thetam3)*x[4*n_nodes:5*n_nodes] +r_gb*(thetam1+ thetam2+ thetam3)*(x[:n_nodes]+ x[n_nodes:2*n_nodes]+ x[2*n_nodes:3*n_nodes])  \
                                + (r_bb/2)*thetam2*(x[3*n_nodes: 4*n_nodes]+ x[5*n_nodes: 6*n_nodes]) + (r_bb/2)*thetam1*(x[5*n_nodes: 6*n_nodes]) + (r_bb/2)*thetam3*(x[3*n_nodes: 4*n_nodes])\
                                    + (p_bb/2)*(x[3*n_nodes: 4*n_nodes]+x[5*n_nodes: 6*n_nodes]) 
    dx[5*n_nodes:6*n_nodes] = -(3*p_bg+p_bb)*x[5*n_nodes:6*n_nodes] - (r_bb*thetam1 +r_bb*thetam2)*x[5*n_nodes:6*n_nodes] + r_gb*(thetam1+ thetam2+ thetam3)*(x[:n_nodes]+ x[n_nodes:2*n_nodes]+ x[2*n_nodes:3*n_nodes])  \
                                + (r_bb/2)*thetam3*(x[3*n_nodes: 4*n_nodes]+ x[4*n_nodes: 5*n_nodes]) + (r_bb/2)*thetam1*(x[4*n_nodes: 5*n_nodes]) + (r_bb/2)*thetam2*(x[3*n_nodes: 4*n_nodes]) \
                                    + (p_bb/2)*(x[3*n_nodes: 4*n_nodes]+ x[4*n_nodes: 5*n_nodes])
    return dx
def cal_B_largest(G, p_bg, p_bb, r_bb, r_gb):
    A = nx.adjacency_matrix(G).T
    n, m =A.get_shape()
    B = vstack([hstack([(r_gb)*A - (3*p_bg+p_bb-2)*csr_matrix(np.eye(n)), (r_gb)*A + (p_bb/2)*csr_matrix(np.eye(n)), (r_gb)*A + (p_bb/2)*csr_matrix(np.eye(n))]), 
            hstack([(r_gb)*A + (p_bb/2)*csr_matrix(np.eye(n)),  (r_gb)*A - (3*p_bg+p_bb-2)*csr_matrix(np.eye(n)), (r_gb)*A + (p_bb/2)*csr_matrix(np.eye(n))]),
            hstack([(r_gb)*A + (p_bb/2)*csr_matrix(np.eye(n)), (r_gb)*A + (p_bb/2)*csr_matrix(np.eye(n)), (r_gb)*A - (3*p_bg+p_bb-2)*csr_matrix(np.eye(n))])])
    evals, x_vec = largest_eigsh(B, 1, which='LM')
    return evals-2

def init_sample(n_nodes, percent):
    x = np.zeros(6*n_nodes)
    selected_indices = np.random.choice(range(n_nodes), size=int(n_nodes*percent), replace=False)
    for i in range(n_nodes):
        if i in selected_indices:
            numbers = [4, 5, 6]
            state = np.random.choice(numbers)
            x[(state-1)*n_nodes + i] = 1
        else:
            numbers = [1,2,3]
            state = np.random.choice(numbers)
            x[(state-1)*n_nodes + i] = 1
    return x


def discrete(y):
    all_counts = []
    samples = np.arange(0,y.shape[0],5)
    y = y[samples]
    numbers = [i for i in range(2)]
    for i in range(y.shape[0]):
        counts = [0 for _ in range(2)]
        yr = y[i].reshape(6,-1)

        probs1 = yr[3:,:].sum(axis = 0)
        for j in range(y.shape[1]//6):
            a = np.random.choice(numbers, p=(1 - probs1[j], probs1[j]))
            counts[a] += 1
        all_counts.append(counts)
    return np.array(all_counts)

def average(data):
    b = Parallel(n_jobs=-1)(delayed(discrete)(data) for i in range(10))
    b = np.array(b)
    return b.mean(axis = 0)

def simulation2(G, p_bg, p_bb, r_bb, r_gb, percent, name):
    n_nodes = G.number_of_nodes()
    t_span = (0, 40)
    t_eval = np.linspace(0, 40, 60)
    A = nx.adjacency_matrix(G).T
    n_nodes, _ = A.get_shape()
    y0 = init_sample(n_nodes, percent)
    print(sum(y0))
    y = solve_ivp(equations, t_span, y0, t_eval=t_eval, args=(p_bg, p_bb, r_bb, r_gb,A))
    return y
p_bg = 0.3

if __name__ == '__main__':
    data1 = "./graphs/p2p-Gnutella05.txt"
    data2 = "./graphs/Wiki-Vote.txt"
    data3 = "./graphs/as-caida20040105.txt"
    
    G1 = read_txt_direct(data1)
    G2 = read_txt_direct(data2)
    G3 = load_as_caida(data3)
    p_bg = 0.7
    p_bb = 0.1
    r_bb = 0.1
    r_gb = 0.13
    cal_B_largest(G1, p_bg, p_bb, r_bb, r_gb)
    rs = np.linspace(0.03, 0.23, 11)
    barxinfty = []
    lambdaB = []
    for r_gb in tqdm(rs):
        print(r_gb)
        y = simulation2(G1, p_bg, p_bb, r_bb, r_gb, 0.002, '')
        lambdaB.append(cal_B_largest(G1, p_bg, p_bb, r_bb, r_gb)[0])
        barxinfty.append((y.y[-3*8846:,:].sum(axis = 0)/8846).mean())
    results1 = np.vstack([lambdaB, barxinfty])
    np.save('./data/sensity/Gnutellagamma.npy', results1)