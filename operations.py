from myModelConfig import Configuration
from scipy.integrate import odeint
from mysir import SIRModel
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.sparse.linalg import eigsh as largest_eigsh
#抽取txt中的数据
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
import gc
import copy

def read_txt_undirect(data):
    g = nx.read_edgelist(data, create_using=nx.Graph())
    return g
def read_txt_direct(data):
    g = nx.read_edgelist(data, create_using=nx.DiGraph())
    return g
model_parameters = {"gamma":{
    "gamma_1m1m1": 0.05, "gamma_1m1m2": 0, "gamma_1m1m3":0,
    "gamma_1m2m1": 0.05, "gamma_1m2m2": 0.05, "gamma_1m2m3":0,
    "gamma_1m3m1": 0, "gamma_1m3m2": 0, "gamma_1m3m3": 0,
    "gamma_2m1m1": 0, "gamma_2m1m2": 0.05, "gamma_2m1m3":0,
    "gamma_2m2m1":0, "gamma_2m2m2": 0.05, "gamma_2m2m3": 0,
    "gamma_2m3m1": 0, "gamma_2m3m2": 0, "gamma_2m3m3":0,
    "gamma_3m1m1": 0,  "gamma_3m1m2": 0, "gamma_3m1m3": 0, 
    "gamma_3m2m1": 0,  "gamma_3m2m2": 0, "gamma_3m2m3":0,
    "gamma_3m3m1":0, "gamma_3m3m2": 0, "gamma_3m3m3": 0,
    "gamma_m1m2m2": 0.03, "gamma_m1m2m3": 0, 
    "gamma_m1m3m2":0, "gamma_m1m3m3":0,
    "gamma_m2m1m1": 0.03, "gamma_m2m1m3": 0,
    "gamma_m2m3m3": 0, "gamma_m2m3m1": 0,
    "gamma_m3m1m1":0, "gamma_m3m1m2": 0,
    "gamma_m3m2m1":0, "gamma_m3m2m2":0},
 "beta":{
            "p_12":0.2, "p_13":0,
            "p_23":0, "p_21": 0.3, "p_23": 0,
            "p_31":0, "p_32": 0, 
            "p_m11":0.3, "p_m12": 0.3, "p_m13":0,
            "p_m21": 0.3, "p_m22": 0.3, "p_m23": 0, 
            "p_m31": 0, "p_m32":0, "p_m33": 0,
            "p_m1m2": 0.1, "p_m1m3":0, 
            "p_m2m1": 0.1, "p_m2m3": 0,
            "p_m3m1": 0, "p_m3m2": 0
},
 "infection_rate":{"3_infected":0, "2_infected":0.4, "1_infected":0.4, "m1_infected":0.1, "m2_infected":0.1, "m3_infected":0}                
} 

def set_config(model, model_parameters):
   config = Configuration()
    # 加入参数
   for k in model_parameters["gamma"].keys():
      config.add_model_parameter(k, model_parameters["gamma"][k])
   for k in model_parameters["beta"].keys():
      config.add_model_parameter(k, model_parameters["beta"][k])
   for k in model_parameters["infection_rate"].keys():
      config.add_model_parameter(k, model_parameters["infection_rate"][k])
   model.set_initial_status(config)
   return model 
# model = set_config(model, model_parameters)
# iterations = model.iteration_bunch(1000)
# trends = model.build_trends(iterations)

####################################计算最大特征值####################################################

def cal_largest_eigen(model_parameters, g, n_state=2, m_state=2):
    p12 = model_parameters["beta"]["p_12"]
    p13 = model_parameters["beta"]["p_13"]
    p21 = model_parameters["beta"]["p_21"]
    p31 = model_parameters["beta"]["p_31"]
    p32 = model_parameters["beta"]["p_32"]
    p23 = model_parameters["beta"]["p_23"]
    pm11 = model_parameters["beta"]["p_m11"]
    pm12 = model_parameters["beta"]["p_m12"]
    pm13 = model_parameters["beta"]["p_m13"]
    pm21 = model_parameters["beta"]["p_m21"]
    pm22 = model_parameters["beta"]["p_m22"]
    pm23 = model_parameters["beta"]["p_m23"]
    pm31 = model_parameters["beta"]["p_m31"]
    pm32 = model_parameters["beta"]["p_m32"]
    pm33 = model_parameters["beta"]["p_m33"]
    pm1m2 = model_parameters["beta"]["p_m1m2"]
    pm1m3 = model_parameters["beta"]["p_m1m3"]
    pm2m1 = model_parameters["beta"]["p_m2m1"]
    pm2m3 = model_parameters["beta"]["p_m2m3"]
    pm3m1 = model_parameters["beta"]["p_m3m1"]
    pm3m2 = model_parameters["beta"]["p_m3m2"]
    gamma_1m1m1 = model_parameters["gamma"]["gamma_1m1m1"]
    gamma_1m1m2 = model_parameters["gamma"]["gamma_1m1m2"]
    gamma_1m1m3 = model_parameters["gamma"]["gamma_1m1m3"]
    gamma_1m2m1 = model_parameters["gamma"]["gamma_1m2m1"]
    gamma_1m2m2 = model_parameters["gamma"]["gamma_1m2m2"]
    gamma_1m2m3 = model_parameters["gamma"]["gamma_1m2m3"]
    gamma_1m3m1 = model_parameters["gamma"]["gamma_1m3m1"]
    gamma_1m3m2 = model_parameters["gamma"]["gamma_1m3m2"]
    gamma_1m3m3 = model_parameters["gamma"]["gamma_1m3m3"]
    gamma_2m1m1 = model_parameters["gamma"]["gamma_2m1m1"]
    gamma_2m1m2 = model_parameters["gamma"]["gamma_2m1m2"]
    gamma_2m1m3 = model_parameters["gamma"]["gamma_2m1m3"]
    gamma_2m2m1 = model_parameters["gamma"]["gamma_2m2m1"]
    gamma_2m2m2 = model_parameters["gamma"]["gamma_2m2m2"]
    gamma_2m2m3 = model_parameters["gamma"]["gamma_2m2m3"]
    gamma_2m3m1 = model_parameters["gamma"]["gamma_2m3m1"]
    gamma_2m3m2 = model_parameters["gamma"]["gamma_2m3m2"]
    gamma_2m3m3 = model_parameters["gamma"]["gamma_2m3m3"]
    gamma_3m1m1 = model_parameters["gamma"]["gamma_3m1m1"]
    gamma_3m1m2 = model_parameters["gamma"]["gamma_3m1m2"]
    gamma_3m1m3 = model_parameters["gamma"]["gamma_3m1m3"]
    gamma_3m2m1 = model_parameters["gamma"]["gamma_3m2m1"]
    gamma_3m2m2 = model_parameters["gamma"]["gamma_3m2m2"]
    gamma_3m2m3 = model_parameters["gamma"]["gamma_3m2m3"]
    gamma_3m3m1 = model_parameters["gamma"]["gamma_3m3m1"]
    gamma_3m3m2 = model_parameters["gamma"]["gamma_3m3m2"]
    gamma_3m3m3 = model_parameters["gamma"]["gamma_3m3m3"]
    gamma_m2m1m1 = model_parameters["gamma"]["gamma_m2m1m1"]
    gamma_m2m1m3 = model_parameters["gamma"]["gamma_m2m1m3"]
    gamma_m2m3m1 = model_parameters["gamma"]["gamma_m2m3m1"]
    gamma_m2m3m3 = model_parameters["gamma"]["gamma_m2m3m3"]
    gamma_m1m2m2 = model_parameters["gamma"]["gamma_m1m2m2"]
    gamma_m1m2m3 = model_parameters["gamma"]["gamma_m1m2m3"]
    gamma_m1m3m2 = model_parameters["gamma"]["gamma_m1m3m2"]
    gamma_m1m3m3 = model_parameters["gamma"]["gamma_m1m3m3"]
    gamma_m3m1m2 = model_parameters["gamma"]["gamma_m3m1m2"]
    gamma_m3m1m1 = model_parameters["gamma"]["gamma_m3m1m1"]
    gamma_m3m2m1 = model_parameters["gamma"]["gamma_m3m2m1"]
    gamma_m3m2m2 = model_parameters["gamma"]["gamma_m3m2m2"]
    A = nx.adjacency_matrix(g).T
    n, m =A.get_shape()
    if n_state == 3:
        a2 = np.array([[-p31-p32,p23,p13],[p32, -p23-p21,p12], [1,1,1]])
        b2 = np.array([0,0,1])
        c2 = np.linalg.solve(a2,b2)
        x3, x2, x1 = c2[0], c2[1], c2[2]
        B = vstack([hstack([(x1*gamma_1m1m1+x2*gamma_2m1m1+ x3*gamma_3m1m1)*A - (pm11+pm12+pm13+pm1m2-2)*csr_matrix(np.eye(n)), (x1*gamma_1m2m1 + x2*gamma_2m2m1 + x3*gamma_3m2m1)*A + pm2m1*csr_matrix(np.eye(n))]), 
        hstack([(x1*gamma_1m1m2 + x2*gamma_2m1m2 + x3*gamma_3m1m2)*A + pm1m2*csr_matrix(np.eye(n)), (x1*gamma_1m2m2+x2*gamma_2m2m2+ x3*gamma_3m2m2)*A - (pm21+pm22+pm23+pm2m1-2)*csr_matrix(np.eye(n))])])
        evals, x_vec = largest_eigsh(B, 1, which='LM')
        _, y_vec = largest_eigsh(B.T, 1, which='LM')
    elif n_state == 2:
        x2 = p12/(p12+p21)
        x1 = p21/(p12+p21)
        if m_state == 2:
            B = vstack([hstack([(x1*gamma_1m1m1+x2*gamma_2m1m1)*A - (pm11+pm12+pm1m2+pm1m3-2)*csr_matrix(np.eye(n)), (x1*gamma_1m2m1 + x2*gamma_2m2m1)*A + pm2m1*csr_matrix(np.eye(n))]), 
            hstack([(x1*gamma_1m1m2 + x2*gamma_2m1m2)*A + pm1m2*csr_matrix(np.eye(n)), (x1*gamma_1m2m2+x2*gamma_2m2m2)*A  - (pm21+pm22+pm2m1+pm2m3-2)*csr_matrix(np.eye(n))])])
            evals, x_vec = largest_eigsh(B, 1, which='LM')
            _, y_vec = largest_eigsh(B.T, 1, which='LM')
        else:
            B = vstack([hstack([(x1*gamma_1m1m1+x2*gamma_2m1m1 )*A - (pm11+pm12+pm13+pm1m2+pm1m3-2)*csr_matrix(np.eye(n)), (x1*gamma_1m2m1 + x2*gamma_2m2m1)*A+pm2m1*csr_matrix(np.eye(n)), (x1*gamma_1m3m1 + x2*gamma_2m3m1)*A + pm3m1*csr_matrix(np.eye(n))]), 
            hstack([(x1*gamma_1m1m2 + x2*gamma_2m1m2)*A + pm1m2*csr_matrix(np.eye(n)),  (x1*gamma_1m2m2+x2*gamma_2m2m2)*A  - (pm21+pm22+pm23+pm2m1+pm2m3-2)*csr_matrix(np.eye(n)), (x1*gamma_1m3m2 + x2*gamma_2m3m2)*A + pm3m2*csr_matrix(np.eye(n))]),
            hstack([(x1*gamma_1m1m3 + x2*gamma_2m1m3)*A + pm1m3*csr_matrix(np.eye(n)), (x1*gamma_1m2m3+x2*gamma_2m2m3)*A + pm2m3*csr_matrix(np.eye(n)), (x1*gamma_1m3m3 + x2*gamma_2m3m3)*A- (pm31+pm32+pm33+pm3m2+pm3m1-2)*csr_matrix(np.eye(n))])])
            evals, x_vec = largest_eigsh(B, 1, which='LM')
            _, y_vec = largest_eigsh(B.T, 1, which='LM')
    del B 
    gc.collect()
    return evals-2, x_vec, y_vec


#####################################计算B0的最大特征值#############################################
def cal_B0_largest_eigen(model_parameters, g, n_state=2, m_state=2):
    p12 = model_parameters["beta"]["p_12"]
    p13 = model_parameters["beta"]["p_13"]
    p21 = model_parameters["beta"]["p_21"]
    p31 = model_parameters["beta"]["p_31"]
    p32 = model_parameters["beta"]["p_32"]
    p23 = model_parameters["beta"]["p_23"]
    pm11 = model_parameters["beta"]["p_m11"]
    pm12 = model_parameters["beta"]["p_m12"]
    pm13 = model_parameters["beta"]["p_m13"]
    pm21 = model_parameters["beta"]["p_m21"]
    pm22 = model_parameters["beta"]["p_m22"]
    pm23 = model_parameters["beta"]["p_m23"]
    pm31 = model_parameters["beta"]["p_m31"]
    pm32 = model_parameters["beta"]["p_m32"]
    pm33 = model_parameters["beta"]["p_m33"]
    pm1m2 = model_parameters["beta"]["p_m1m2"]
    pm1m3 = model_parameters["beta"]["p_m1m3"]
    pm2m1 = model_parameters["beta"]["p_m2m1"]
    pm2m3 = model_parameters["beta"]["p_m2m3"]
    pm3m1 = model_parameters["beta"]["p_m3m1"]
    pm3m2 = model_parameters["beta"]["p_m3m2"]
    gamma_1m1m1 = model_parameters["gamma"]["gamma_1m1m1"]
    gamma_1m1m2 = model_parameters["gamma"]["gamma_1m1m2"]
    gamma_1m1m3 = model_parameters["gamma"]["gamma_1m1m3"]
    gamma_1m2m1 = model_parameters["gamma"]["gamma_1m2m1"]
    gamma_1m2m2 = model_parameters["gamma"]["gamma_1m2m2"]
    gamma_1m2m3 = model_parameters["gamma"]["gamma_1m2m3"]
    gamma_1m3m1 = model_parameters["gamma"]["gamma_1m3m1"]
    gamma_1m3m2 = model_parameters["gamma"]["gamma_1m3m2"]
    gamma_1m3m3 = model_parameters["gamma"]["gamma_1m3m3"]
    gamma_2m1m1 = model_parameters["gamma"]["gamma_2m1m1"]
    gamma_2m1m2 = model_parameters["gamma"]["gamma_2m1m2"]
    gamma_2m1m3 = model_parameters["gamma"]["gamma_2m1m3"]
    gamma_2m2m1 = model_parameters["gamma"]["gamma_2m2m1"]
    gamma_2m2m2 = model_parameters["gamma"]["gamma_2m2m2"]
    gamma_2m2m3 = model_parameters["gamma"]["gamma_2m2m3"]
    gamma_2m3m1 = model_parameters["gamma"]["gamma_2m3m1"]
    gamma_2m3m2 = model_parameters["gamma"]["gamma_2m3m2"]
    gamma_2m3m3 = model_parameters["gamma"]["gamma_2m3m3"]
    gamma_3m1m1 = model_parameters["gamma"]["gamma_3m1m1"]
    gamma_3m1m2 = model_parameters["gamma"]["gamma_3m1m2"]
    gamma_3m1m3 = model_parameters["gamma"]["gamma_3m1m3"]
    gamma_3m2m1 = model_parameters["gamma"]["gamma_3m2m1"]
    gamma_3m2m2 = model_parameters["gamma"]["gamma_3m2m2"]
    gamma_3m2m3 = model_parameters["gamma"]["gamma_3m2m3"]
    gamma_3m3m1 = model_parameters["gamma"]["gamma_3m3m1"]
    gamma_3m3m2 = model_parameters["gamma"]["gamma_3m3m2"]
    gamma_3m3m3 = model_parameters["gamma"]["gamma_3m3m3"]
    gamma_m2m1m1 = model_parameters["gamma"]["gamma_m2m1m1"]
    gamma_m2m1m3 = model_parameters["gamma"]["gamma_m2m1m3"]
    gamma_m2m3m1 = model_parameters["gamma"]["gamma_m2m3m1"]
    gamma_m2m3m3 = model_parameters["gamma"]["gamma_m2m3m3"]
    gamma_m1m2m2 = model_parameters["gamma"]["gamma_m1m2m2"]
    gamma_m1m2m3 = model_parameters["gamma"]["gamma_m1m2m3"]
    gamma_m1m3m2 = model_parameters["gamma"]["gamma_m1m3m2"]
    gamma_m1m3m3 = model_parameters["gamma"]["gamma_m1m3m3"]
    gamma_m3m1m2 = model_parameters["gamma"]["gamma_m3m1m2"]
    gamma_m3m1m1 = model_parameters["gamma"]["gamma_m3m1m1"]
    gamma_m3m2m1 = model_parameters["gamma"]["gamma_m3m2m1"]
    gamma_m3m2m2 = model_parameters["gamma"]["gamma_m3m2m2"]
    A = nx.adjacency_matrix(g).T
    n, m =A.get_shape()
    if m_state == 2:
        B0 = vstack([hstack([(gamma_1m1m1+gamma_2m1m1 + gamma_m2m1m1)*A.T - (pm11+pm12+pm1m2+pm1m3-2)*csr_matrix(np.eye(n)), (gamma_1m2m1 + gamma_2m2m1)*A.T + pm2m1*csr_matrix(np.eye(n))]), 
        hstack([(gamma_1m1m2 + gamma_2m1m2)*A.T + pm1m2*csr_matrix(np.eye(n)), (gamma_1m2m2+gamma_2m2m2+gamma_m1m2m2)*A.T  - (pm21+pm22+pm2m1+pm2m3-2)*csr_matrix(np.eye(n))])])
    else:
        B0 = vstack([hstack([(gamma_1m1m1+gamma_2m1m1+ gamma_m2m1m1 )*A - (pm11+pm12+pm13+pm1m2+pm1m3-2)*csr_matrix(np.eye(n)), (gamma_1m2m1 + gamma_2m2m1)*A+pm2m1*csr_matrix(np.eye(n)), (gamma_1m3m1 + gamma_2m3m1)*A + pm3m1*csr_matrix(np.eye(n))]), 
            hstack([(gamma_1m1m2 + gamma_2m1m2)*A + pm1m2*csr_matrix(np.eye(n)),  (gamma_1m2m2+gamma_2m2m2+gamma_m1m2m2)*A  - (pm21+pm22+pm23+pm2m1+pm2m3-2)*csr_matrix(np.eye(n)), (gamma_1m3m2 + gamma_2m3m2)*A + pm3m2*csr_matrix(np.eye(n))]),
            hstack([(gamma_1m1m3 + gamma_2m1m3)*A + pm1m3*csr_matrix(np.eye(n)), (gamma_1m2m3+gamma_2m2m3)*A + pm2m3*csr_matrix(np.eye(n)), (gamma_1m3m3 + gamma_2m3m3)*A- (pm31+pm32+pm33+pm3m2+pm3m1-2)*csr_matrix(np.eye(n))])])
    evals, x_vec = largest_eigsh(B0, 1, which='LM')
    _, y_vec = largest_eigsh(B0.T, 1, which='LM')
    del B0 
    gc.collect()
    return evals-2, x_vec, y_vec

#####################################改变参数############################################

def change_bad_to_good(ep, model_parameters):
    model_p = copy.deepcopy(model_parameters)
    model_p["beta"]["p_m11"] = ep
    model_p["beta"]["p_m12"] = ep
    model_p["beta"]["p_m21"] = ep
    model_p["beta"]["p_m22"] = ep
    return model_p, 2, 2

def change_gamma(ep, model_parameters):
    model_p = copy.deepcopy(model_parameters)
    defa = model_parameters["gamma"]["gamma_1m1m1"]
    model_p["gamma"]["gamma_1m1m1"] = defa-ep
    model_p["gamma"]["gamma_1m1m2"] = defa-ep
    model_p["gamma"]["gamma_1m2m1"] = defa-ep 
    model_p["gamma"]["gamma_1m2m2"] = defa-ep
    model_p["gamma"]["gamma_2m1m1"] = defa-ep
    model_p["gamma"]["gamma_2m1m2"] = defa-ep
    model_p["gamma"]["gamma_2m2m1"] = defa-ep
    model_p["gamma"]["gamma_2m2m2"] = defa-ep
    return model_p, 2, 2

def add_bad_mtd(ep, model_parameters):
    model_p = copy.deepcopy(model_parameters)
    gamma = model_parameters["gamma"]["gamma_1m1m1"]
    model_p["beta"]["p_m3m1"] = 0
    model_p["beta"]["p_m3m2"] = 0
    model_p["beta"]["p_m31"] = ep  
    model_p["beta"]["p_m32"] = ep 
    model_p["beta"]["p_m1m3"] = ep/2
    model_p["beta"]["p_m2m3"] = ep/2
    model_p["gamma"]["gamma_1m1m3"] = gamma[0]*ep*2
    model_p["gamma"]["gamma_1m2m3"] = gamma[1]*ep*2
    model_p["gamma"]["gamma_2m1m3"] = gamma[2]*ep*2
    model_p["gamma"]["gamma_2m2m3"] = gamma[3]*ep*2
    model_p["gamma"]["gamma_2m2m2"] = gamma[0]*(1-2*ep)
    model_p["gamma"]["gamma_2m1m1"] = gamma[1]*(1-2*ep)
    model_p["gamma"]["gamma_1m2m2"] = gamma[2]*(1-2*ep)
    model_p["gamma"]["gamma_1m1m1"] = gamma[3]*(1-2*ep)
    model_p["gamma"]["gamma_1m3m1"] = 0
    model_p["gamma"]["gamma_1m3m2"] = 0
    return model_p, 2, 3


def add_bad_active(ep, model_parameters):
    model_p = copy.deepcopy(model_parameters)
    model_p["beta"]["p_m3m1"] = 0
    model_p["beta"]["p_m3m2"] = 0
    model_p["beta"]["p_m31"] = ep  
    model_p["beta"]["p_m32"] = ep
    model_p["beta"]["p_m1m3"] = ep/2
    model_p["beta"]["p_m2m3"] = ep/2
    model_p["gamma"]["gamma_1m1m3"] = 0
    model_p["gamma"]["gamma_1m2m3"] = 0
    model_p["gamma"]["gamma_2m1m3"] = 0
    model_p["gamma"]["gamma_2m2m3"] = 0
    model_p["gamma"]["gamma_2m2m2"] = 0
    model_p["gamma"]["gamma_2m1m1"] = 0
    model_p["gamma"]["gamma_1m2m2"] = 0
    model_p["gamma"]["gamma_1m1m1"] = 0
    model_p["gamma"]["gamma_m1m3m3"] = ep/2
    model_p["gamma"]["gamma_m2m3m3"] = ep/2
    return model_p, 2, 3   


#####################################模拟#########################################
def general_simualtion(model_parameters, g, epsilon, change_parameters):
    model = SIRModel(g)
    results = Parallel(n_jobs=5)(delayed(single_simulation)(ep, model, model_parameters, change_parameters) for ep in tqdm(epsilon))
    return results

def single_simulation(ep, model, model_parameters, change_parameters):
    model_para_new,_,_ = change_parameters(ep, model_parameters)
    model_ep = set_config(model, model_para_new)
    iterations = model_ep.iteration_bunch(600)
    trends = model.build_trends(iterations)
    return trends


###################################计算特征值######################################################
def cal_eigen(model_parameters, g, epsilon, change_parameters, task):
    assert task in ['b2g', 'gamma', 'mtd', 'active']
    eigen = []
    eigen_est = []
    if task == 'b2g':
        for ep in range(epsilon):
            model_para_new, n_state, m_state = change_parameters(ep, model_parameters)
            eig, _, _ = cal_largest_eigen(model_para_new, g, n_state, m_state)
            eigen.append(eig)
            eigen_est.append(-2*ep)
    elif task == 'gamma':
        model_para_new, n_state, m_state = change_parameters(0, model_parameters)
        A = nx.adjacency_matrix(g)
        ev, x_vec, y_vec = cal_largest_eigen(model_para_new, g, n_state=2, m_state=2)
        n_nodes = len(g.nodes)
        delta_eigen = csr_matrix.dot(csr_matrix.dot((y_vec[:n_nodes]+y_vec[n_nodes:]).T, A), x_vec[:n_nodes] + x_vec[n_nodes:])/csr_matrix.dot(y_vec.T,x_vec)[0][0]
        del A
        gc.collect()
        for ep in range(epsilon):
            model_para_new, n_state, m_state = change_parameters(ep, model_parameters)
            eig, _, _ = cal_largest_eigen(model_para_new, g, n_state, m_state)
            eigen.append(eig)
            eigen_est.append(-ep*delta_eigen[0][0])
    elif task == 'mtd':
        model_para_new, n_state, m_state = change_parameters(0, model_parameters)
        A = nx.adjacency_matrix(g)
        ev, x_vec, y_vec = cal_largest_eigen(model_para_new, g, n_state=2, m_state=2)
        n_nodes = len(g.nodes)
        delta_eigen = (csr_matrix.dot(csr_matrix.dot(y_vec[:n_nodes].T, A), x_vec[:n_nodes]) + csr_matrix.dot(csr_matrix.dot(y_vec[n_nodes:].T, A), x_vec[n_nodes:]))/csr_matrix.dot(y_vec.T,x_vec)[0][0]
        del A
        gc.collect()
        for ep in range(epsilon):
            model_para_new, n_state, m_state = change_parameters(ep, model_parameters)
            eig, _, _ = cal_largest_eigen(model_para_new, g, n_state, m_state)
            eigen.append(eig)
            eigen_est.append(-2*ep*delta_eigen[0][0]-ep/2)
    elif task =='active':
        for ep in range(epsilon):
            model_para_new, n_state, m_state = change_parameters(ep, model_parameters)
            eig, _, _ = cal_largest_eigen(model_para_new, g, n_state, m_state)
            eigen.append(eig)
            eigen_est.append(-ep/2)
    return eigen, eigen_est
    
#############################################数值计算#############################################################################
               
               
               
               
                   
               
####################################保存数据生成图像################################################################################
def single_initial(ep, model, model_parameters, n_nodes):
    model_parameters["infection_rate"]["2_infected"] = (1-2*ep)/2
    model_parameters["infection_rate"]["1_infected"] = (1-2*ep)/2
    model_parameters["infection_rate"]["m1_infected"]= ep
    model_parameters["infection_rate"]["m2_infected"] = ep
    model = set_config(model, model_parameters)
    iterations = model.iteration_bunch(800)
    trends = model.build_trends(iterations)
    tds = []
    for i in range(-50,0):
        tds.append((trends[0]["trends"]["node_count"][-2][i]+trends[0]["trends"]["node_count"][-2][i])/n_nodes)
    tds = np.array(tds)
    simus = np.array([tds.mean(), tds.max()-tds.mean(), tds.mean()-tds.min()])
    return simus


def change_initial(model_parameters, g, n_state=2, m_state=2, epislon = np.linspace(0,0.3,11)):
    model = SIRModel(g)
    n_nodes = len(g.nodes)
    results = Parallel(n_jobs=5)(delayed(single_initial)(ep, model, model_parameters, n_nodes) for ep in tqdm(epislon))
    return results


#############################################探究初值的影响###########################################################################

# def cal_badB(ep, g):
#     model_parameters["beta"]["p_m11"] = ep
#     model_parameters["beta"]["p_m12"] = ep
#     model_parameters["beta"]["p_m21"] = ep
#     model_parameters["beta"]["p_m22"] = ep
#     model = set_config(model, model_parameters)
#     iterations = model.iteration_bunch(800)
#     trends = model.build_trends(iterations)
#     # simus.append([tds.mean(), tds.amx()-tds.mean(), tds.mean()-tds.min()])
#     eigenval, _, _ = cal_largest_eigen(model_parameters, g, n_state=2, m_state=2)
#     # eigens.append(eigenval)
#     delta_eigen = -2*ep
#     return eigenval, delta_eigen


def single_bad(ep, model, model_parameters, n_nodes, x_vec, y_vec, g):
    model_parameters["beta"]["p_m11"] = ep
    model_parameters["beta"]["p_m12"] = ep
    model_parameters["beta"]["p_m21"] = ep
    model_parameters["beta"]["p_m22"] = ep
    model = set_config(model, model_parameters)
    iterations = model.iteration_bunch(800)
    trends = model.build_trends(iterations)
    tds = []
    for i in range(-50,0):
        tds.append((trends[0]["trends"]["node_count"][-2][i]+trends[0]["trends"]["node_count"][-2][i])/n_nodes)
    tds = np.array(tds)
    simus = np.array([tds.mean(), tds.max()-tds.mean(), tds.mean()-tds.min()])
    # simus.append([tds.mean(), tds.amx()-tds.mean(), tds.mean()-tds.min()])
    eigenval, _, _ = cal_largest_eigen(model_parameters, g, n_state=2, m_state=2)
    # eigens.append(eigenval)
    delta_eigen = -2*ep
    return simus, eigenval, delta_eigen


def change_bad_to_good(model_parameters, g, n_state=2, m_state=2, epislon = np.linspace(0,0.3,11)):
    model = SIRModel(g)
    n_nodes = len(g.nodes)
    model_parameters["beta"]["p_m11"] = 0
    model_parameters["beta"]["p_m12"] = 0
    model_parameters["beta"]["p_m21"] = 0
    model_parameters["beta"]["p_m22"] = 0
    ev, x_vec, y_vec = cal_largest_eigen(model_parameters, g, n_state=2, m_state=2)
    results = Parallel(n_jobs=4)(delayed(single_bad)(ep, model, model_parameters, n_nodes, x_vec, y_vec, g) for ep in tqdm(epislon))
    simu, eigen, eigen_est = process_result(results)
    return simu, eigen, eigen_est
    
########################################减少感染率############################################################################
    
def change_infectious_to_good(model_parameters, g, defa, epislon):
    model = SIRModel(g)
    n_nodes = len(g.nodes)
    model_parameters["gamma"]["gamma_1m1m1"] = defa
    model_parameters["gamma"]["gamma_1m1m2"] = defa
    model_parameters["gamma"]["gamma_1m2m1"] = defa  
    model_parameters["gamma"]["gamma_1m2m2"] = defa 
    model_parameters["gamma"]["gamma_2m1m1"] = defa
    model_parameters["gamma"]["gamma_2m1m2"] = defa
    model_parameters["gamma"]["gamma_2m2m1"] = defa
    model_parameters["gamma"]["gamma_2m2m2"] = defa
    ev, x_vec, y_vec = cal_largest_eigen(model_parameters, g, n_state=2, m_state=2)
    result = Parallel(n_jobs=4)(delayed(single_infect)(ep, model, model_parameters, n_nodes, x_vec, y_vec, g, defa) for ep in tqdm(epislon))
    simu, eigen, eigen_est = process_result(result)
    return simu, eigen, eigen_est

def single_infect(ep, model, model_parameters, n_nodes, x_vec, y_vec, g, default):
    model_parameters["gamma"]["gamma_1m1m1"] = default - ep
    model_parameters["gamma"]["gamma_1m1m2"] = default - ep
    model_parameters["gamma"]["gamma_1m2m1"] = default - ep
    model_parameters["gamma"]["gamma_1m2m2"] = default - ep
    model_parameters["gamma"]["gamma_2m1m1"] = default - ep
    model_parameters["gamma"]["gamma_2m1m2"] = default - ep
    model_parameters["gamma"]["gamma_2m2m1"] = default - ep
    model_parameters["gamma"]["gamma_2m2m2"] = default - ep
    model = set_config(model, model_parameters)
    iterations = model.iteration_bunch(600)
    trends = model.build_trends(iterations)
    tds = []
    for i in range(-50,0):
        tds.append((trends[0]["trends"]["node_count"][-2][i]+trends[0]["trends"]["node_count"][-2][i])/n_nodes)
    tds = np.array(tds)
    simus = np.array([tds.mean(), tds.max()-tds.mean(), tds.mean()-tds.min()])
    # simus.append([tds.mean(), tds.amx()-tds.mean(), tds.mean()-tds.min()])
    eigenval, _, _ = cal_largest_eigen(model_parameters, g, n_state=2, m_state=2)
    # eigens.append(eigenval)
    A = nx.adjacency_matrix(g)
    delta = vstack([hstack([-ep*A, -ep*A]),
            hstack([-ep*A, -ep*A])])
    delta_eigen = csr_matrix.dot(csr_matrix.dot(y_vec.T, delta), x_vec)/csr_matrix.dot(y_vec.T,x_vec)[0][0]
    return simus, eigenval, delta_eigen[0][0]




########################################moving target defense#################################################################

def add_bad_mtd(model_parameters, g, epislon = np.linspace(0,0.4,11)):
    model = SIRModel(g)
    n_nodes = len(g.nodes)
    model_parameters["beta"]["p_m3m1"] = 0
    model_parameters["beta"]["p_m3m2"] = 0
    model_parameters["beta"]["p_m31"] = 0  
    model_parameters["beta"]["p_m32"] = 0
    model_parameters["beta"]["p_m1m3"] = 0
    model_parameters["beta"]["p_m2m3"] = 0
    model_parameters["gamma"]["gamma_1m1m3"] = 0
    model_parameters["gamma"]["gamma_1m2m3"] = 0
    model_parameters["gamma"]["gamma_2m1m3"] = 0
    model_parameters["gamma"]["gamma_2m2m3"] = 0
    gamma = (model_parameters["gamma"]["gamma_2m2m2"], model_parameters["gamma"]["gamma_2m1m1"], model_parameters["gamma"]["gamma_1m2m2"], model_parameters["gamma"]["gamma_1m1m1"])
    A = nx.adjacency_matrix(g)
    ev, x_vec, y_vec = cal_largest_eigen(model_parameters, g, n_state=2, m_state=2)
    delta_eigen = csr_matrix.dot(csr_matrix.dot(y_vec[:n_nodes].T, A), x_vec[:n_nodes]) + csr_matrix.dot(csr_matrix.dot(y_vec[n_nodes:].T, A), x_vec[n_nodes:])/csr_matrix.dot(y_vec.T,x_vec)[0][0]
    results = Parallel(n_jobs=4)(delayed(single_bad_mtd)(ep, model, model_parameters, g, n_nodes, gamma, delta_eigen) for ep in tqdm(epislon))
    # simu, eigen, eigen_est = process_result(results)
    return results
                 
                 
def single_bad_mtd(ep, model, model_parameters, g, n_nodes, gamma, delta_eigen):
    model_parameters["beta"]["p_m3m1"] = 0
    model_parameters["beta"]["p_m3m2"] = 0
    model_parameters["beta"]["p_m31"] = ep  
    model_parameters["beta"]["p_m32"] = ep 
    model_parameters["beta"]["p_m1m3"] = ep/2
    model_parameters["beta"]["p_m2m3"] = ep/2
    model_parameters["gamma"]["gamma_1m1m3"] = gamma[0]*ep*2
    model_parameters["gamma"]["gamma_1m2m3"] = gamma[1]*ep*2
    model_parameters["gamma"]["gamma_2m1m3"] = gamma[2]*ep*2
    model_parameters["gamma"]["gamma_2m2m3"] = gamma[3]*ep*2
    model_parameters["gamma"]["gamma_2m2m2"] = gamma[0]*(1-2*ep)
    model_parameters["gamma"]["gamma_2m1m1"] = gamma[1]*(1-2*ep)
    model_parameters["gamma"]["gamma_1m2m2"] = gamma[2]*(1-2*ep)
    model_parameters["gamma"]["gamma_1m1m1"] = gamma[3]*(1-2*ep)
    model_parameters["gamma"]["gamma_1m3m1"] = 0
    model_parameters["gamma"]["gamma_1m3m2"] = 0
    model = set_config(model, model_parameters)
    iterations = model.iteration_bunch(800)
    trends = model.build_trends(iterations)
    tds = trends[0]["trends"]["node_count"]
    eigenval, _, _ = cal_largest_eigen(model_parameters, g, n_state=2, m_state=3)
    # eigens.append(eigenval)
    delta = -2*ep*gamma[0]*delta_eigen - ep/2
    return tds, eigenval, delta


#############################################################active defense########################################################################

def add_bad_active(model_parameters, g, epislon = np.linspace(0,0.4,11)):
    model = SIRModel(g)
    n_nodes = len(g.nodes)
    model_parameters["beta"]["p_m3m1"] = 0
    model_parameters["beta"]["p_m3m2"] = 0
    model_parameters["beta"]["p_m31"] = 0  
    model_parameters["beta"]["p_m32"] = 0
    model_parameters["beta"]["p_m1m3"] = 0
    model_parameters["beta"]["p_m2m3"] = 0
    model_parameters["gamma"]["gamma_1m1m3"] = 0
    model_parameters["gamma"]["gamma_1m2m3"] = 0
    model_parameters["gamma"]["gamma_2m1m3"] = 0
    model_parameters["gamma"]["gamma_2m2m3"] = 0
    
    ev, x_vec, y_vec = cal_largest_eigen(model_parameters, g, n_state=2, m_state=2)
    results = Parallel(n_jobs=4)(delayed(single_bad_active)(ep, model, model_parameters) for ep in tqdm(epislon))
    # e, ee = get_eigen_active(model_parameters, g, epislon)
    return results
                 
                 
def single_bad_active(ep, model, model_parameters):
    model_parameters["beta"]["p_m3m1"] = 0
    model_parameters["beta"]["p_m3m2"] = 0
    model_parameters["beta"]["p_m31"] = ep  
    model_parameters["beta"]["p_m32"] = ep
    model_parameters["beta"]["p_m1m3"] = ep/2
    model_parameters["beta"]["p_m2m3"] = ep/2
    model_parameters["gamma"]["gamma_1m1m3"] = 0
    model_parameters["gamma"]["gamma_1m2m3"] = 0
    model_parameters["gamma"]["gamma_2m1m3"] = 0
    model_parameters["gamma"]["gamma_2m2m3"] = 0
    model_parameters["gamma"]["gamma_2m2m2"] = 0
    model_parameters["gamma"]["gamma_2m1m1"] = 0
    model_parameters["gamma"]["gamma_1m2m2"] = 0
    model_parameters["gamma"]["gamma_1m1m1"] = 0
    model_parameters["gamma"]["gamma_m1m3m3"] = ep/2
    model_parameters["gamma"]["gamma_m2m3m3"] = ep/2
    model = set_config(model, model_parameters)
    iterations = model.iteration_bunch(800)
    trends = model.build_trends(iterations)
    tds = trends[0]["trends"]["node_count"]
    return tds

def get_eigen_active(model_parameters, g, epislon):
    eigen = []
    eigen_est = []
    for ep in epislon:
        model_parameters["beta"]["p_m3m1"] = 0
        model_parameters["beta"]["p_m3m2"] = 0
        model_parameters["beta"]["p_m31"] = ep  
        model_parameters["beta"]["p_m32"] = ep
        model_parameters["beta"]["p_m1m3"] = ep/2
        model_parameters["beta"]["p_m2m3"] = ep/2
        model_parameters["gamma"]["gamma_1m1m3"] = 0
        model_parameters["gamma"]["gamma_1m2m3"] = 0
        model_parameters["gamma"]["gamma_2m1m3"] = 0
        model_parameters["gamma"]["gamma_2m2m3"] = 0
        model_parameters["gamma"]["gamma_2m2m2"] = 0
        model_parameters["gamma"]["gamma_2m1m1"] = 0
        model_parameters["gamma"]["gamma_1m2m2"] = 0
        model_parameters["gamma"]["gamma_1m1m1"] = 0
        model_parameters["gamma"]["gamma_m1m3m3"] = ep/2
        model_parameters["gamma"]["gamma_m2m3m3"] = ep/2
        eigenval, _, _ = cal_largest_eigen(model_parameters, g, n_state=2, m_state=3)
        eigen.append(eigenval)
        eigen_est.append(-ep/2)
    return eigen, eigen_est

def process_result(results):
    simu = []
    eigen = []
    eigen_est = []
    for i in range(11):
        simu.append(results[i][0])
        eigen.append(results[i][1][0])
        eigen_est.append(results[i][2])
    simu = np.array(simu)
    eigen = np.array(eigen)
    eigen_est = np.array(eigen_est)
    eigen_est = eigen_est + eigen[0]
    return simu, eigen, eigen_est

        # eigens_est.append(eigens+ delta_eigen[0][0])


#############################################################数值结果###########################################################################
# 增加gamma
# def trans_quation_fast_map(x, model_parameters, t, G, n_nodes, n_state, m_state):
#     # 兼容无向图和有向图
#     # 两个好节点两个坏节点
#     # 专门写好
#     # 默认参数为D
#     # x 初值 t是时间
#     # 维数为 n,m+n
#     # 理解矩阵的形状 
#     # 不再用矩阵运算, 用矩阵的度去算
#     # x是一个5*n_nodes的向量 顺序依次是 2, 1, -1, -2, -3
#     p12 = model_parameters["beta"]["p_12"]
#     p13 = model_parameters["beta"]["p_13"]
#     p21 = model_parameters["beta"]["p_21"]
#     p31 = model_parameters["beta"]["p_31"]
#     p32 = model_parameters["beta"]["p_32"]
#     p23 = model_parameters["beta"]["p_23"]
#     pm11 = model_parameters["beta"]["p_m11"]
#     pm12 = model_parameters["beta"]["p_m12"]
#     pm13 = model_parameters["beta"]["p_m13"]
#     pm21 = model_parameters["beta"]["p_m21"]
#     pm22 = model_parameters["beta"]["p_m22"]
#     pm23 = model_parameters["beta"]["p_m23"]
#     pm31 = model_parameters["beta"]["p_m31"]
#     pm32 = model_parameters["beta"]["p_m32"]
#     pm33 = model_parameters["beta"]["p_m33"]
#     pm1m2 = model_parameters["beta"]["p_m1m2"]
#     pm1m3 = model_parameters["beta"]["p_m1m3"]
#     pm2m1 = model_parameters["beta"]["p_m2m1"]
#     pm2m3 = model_parameters["beta"]["p_m2m3"]
#     pm3m1 = model_parameters["beta"]["p_m3m1"]
#     pm3m2 = model_parameters["beta"]["p_m3m2"]
#     gamma_1m1m1 = model_parameters["gamma"]["gamma_1m1m1"]
#     gamma_1m1m2 = model_parameters["gamma"]["gamma_1m1m2"]
#     gamma_1m1m3 = model_parameters["gamma"]["gamma_1m1m3"]
#     gamma_1m2m1 = model_parameters["gamma"]["gamma_1m2m1"]
#     gamma_1m2m2 = model_parameters["gamma"]["gamma_1m2m2"]
#     gamma_1m2m3 = model_parameters["gamma"]["gamma_1m2m3"]
#     gamma_1m3m1 = model_parameters["gamma"]["gamma_1m3m1"]
#     gamma_1m3m2 = model_parameters["gamma"]["gamma_1m3m2"]
#     gamma_1m3m3 = model_parameters["gamma"]["gamma_1m3m3"]
#     gamma_2m1m1 = model_parameters["gamma"]["gamma_2m1m1"]
#     gamma_2m1m2 = model_parameters["gamma"]["gamma_2m1m2"]
#     gamma_2m1m3 = model_parameters["gamma"]["gamma_2m1m3"]
#     gamma_2m2m1 = model_parameters["gamma"]["gamma_2m2m1"]
#     gamma_2m2m2 = model_parameters["gamma"]["gamma_2m2m2"]
#     gamma_2m2m3 = model_parameters["gamma"]["gamma_2m2m3"]
#     gamma_2m3m1 = model_parameters["gamma"]["gamma_2m3m1"]
#     gamma_2m3m2 = model_parameters["gamma"]["gamma_2m3m2"]
#     gamma_2m3m3 = model_parameters["gamma"]["gamma_2m3m3"]
#     gamma_3m1m1 = model_parameters["gamma"]["gamma_3m1m1"]
#     gamma_3m1m2 = model_parameters["gamma"]["gamma_3m1m2"]
#     gamma_3m1m3 = model_parameters["gamma"]["gamma_3m1m3"]
#     gamma_3m2m1 = model_parameters["gamma"]["gamma_3m2m1"]
#     gamma_3m2m2 = model_parameters["gamma"]["gamma_3m2m2"]
#     gamma_3m2m3 = model_parameters["gamma"]["gamma_3m2m3"]
#     gamma_3m3m1 = model_parameters["gamma"]["gamma_3m3m1"]
#     gamma_3m3m2 = model_parameters["gamma"]["gamma_3m3m2"]
#     gamma_3m3m3 = model_parameters["gamma"]["gamma_3m3m3"]
#     gamma_m2m1m1 = model_parameters["gamma"]["gamma_m2m1m1"]
#     gamma_m2m1m3 = model_parameters["gamma"]["gamma_m2m1m3"]
#     gamma_m2m3m1 = model_parameters["gamma"]["gamma_m2m3m1"]
#     gamma_m2m3m3 = model_parameters["gamma"]["gamma_m2m3m3"]
#     gamma_m1m2m2 = model_parameters["gamma"]["gamma_m1m2m2"]
#     gamma_m1m2m3 = model_parameters["gamma"]["gamma_m1m2m3"]
#     gamma_m1m3m2 = model_parameters["gamma"]["gamma_m1m3m2"]
#     gamma_m1m3m3 = model_parameters["gamma"]["gamma_m1m3m3"]
#     gamma_m3m1m2 = model_parameters["gamma"]["gamma_m3m1m2"]
#     gamma_m3m1m1 = model_parameters["gamma"]["gamma_m3m1m1"]
#     gamma_m3m2m1 = model_parameters["gamma"]["gamma_m3m2m1"]
#     gamma_m3m2m2 = model_parameters["gamma"]["gamma_m3m2m2"]
#     dxdt = np.zeros_like(x)
#     dxdt[0:n_nodes] = -p21*x[0:n_nodes] + p12*x[n_nodes:2*n_nodes]+pm12*x[2*n_nodes:3*n_nodes] +pm22*x[3*n_nodes:4*n_nodes] + pm32*x[4*n_nodes:5*n_nodes] 
#     dxdt[n_nodes:2*n_nodes] = p21*x[0:n_nodes] -p12*x[n_nodes:2*n_nodes]+pm11*x[2*n_nodes:3*n_nodes] +pm21*x[3*n_nodes:4*n_nodes] + pm31*x[4*n_nodes:5*n_nodes] 
    
#     dxdt[2*n:3*n] = pm1m2*x[n:2*n] - (pm2m1+pm21+pm22)*x[2*n:3*n]
#     for i in G.nodes():
#         num_node = m[i]
#         nei = G.neighbors(i)
#         prod = 1
#         for node in nei:
#             prod*=(1-gamma*x[n:2*n][int(m[node])])
#         dxdt[num_node+n] = (1-prod)*(1-x[num_node]-x[num_node+n] -x[num_node+2*n])-(pm11+pm12+pm1m2)*x[num_node+n]+pm2m1*x[num_node+2*n]
#     # 坏状态-1
#     pass

# def general_map(ep,G,n,x0,t,m):
#     p21 = 0.2
#     p12 = 0.3
#     pm12 = 0.1
#     pm11 = 0.1
#     pm1m2 = 0.2
#     pm22 = 0.2
#     pm21 = 0.2
#     pm2m1 = 0.2
#     gamma = 0.001
#     para = [gamma+ep,p21,p12,pm12,pm11,pm1m2,pm22,pm21,pm2m1]
#     para = (para,n,G,m)
#     result = odeint(trans_quation_fast_map, y0 = x0, t=t, args = para)
#     return result

# def simu_gamma_prall_map(G, type):
#     n = G.number_of_nodes()
#     s = [int(i) for i in G.nodes()]
#     s.sort()
#     map = dict()
#     for i in range(len(s)):
#         map[str(s[i])] = i
#     # A = np.array(nx.adjacency_matrix(G).todense(),dtype='float32')
#     x0 = np.stack([0.2*np.ones(n),0.1*np.ones(n),0.1*np.ones(n)]).reshape(3*n)
#     gap = 200
#     t = np.linspace(0,60,gap)
#     epislon = np.linspace(0,0.02,5)
#     results = Parallel(n_jobs = 5)(delayed(add_gamma_map)(ep,G,n,x0,t,map) for ep in epislon)
#     plt.plot(t[0:gap], results[0][0:gap,n:2*n].sum(axis = 1)/n, label='base',c='r')
#     for i in range(1,5):
#         plt.plot(t[0:gap], results[i][0:gap,n:2*n].sum(axis = 1)/n,c='y')
#     plt.rcParams.update({'font.size':22})
#     plt.legend()
#     # plt.title(type, fontsize = 22)
#     plt.xlabel('t',fontsize = 24)
#     plt.savefig('./picture/picture/openquestion/'+type+'.png')
#     plt.show()



#################################################################生成图像###########################################################
def three_plot(simu, eigen, eigen_est, epislon, num, task):
    names = ["Enron","oregon", "Gnutella"]
    data_path = "C:/Users/xinji/Documents/理论论文/code2/code/models/Simulation/data/"
    with open(data_path+names[num]+task+".pkl", 'wb') as file:
        pickle.dump((simu, eigen, eigen_est), file)
    fig, ax1 = plt.subplots()
    # ax1.xticks(fontsize=20)
    # ax1.yticks(fontsize=20)
    x = [ep for ep in epislon]
    x = [ep for ep in epislon]
    plt.rcParams.update({'font.size':24})
    plt.xticks(fontsize = 26)
    ax1.errorbar(x, y = simu[:,0], yerr = simu[:,1:].T, color="red", label="bad_state")
    ax1.set_xlabel(r"$\epsilon$", fontsize = 26)
    ax1.set_ylabel(r"$\widebar{x_{-1}}+\widebar{x_{-2}}$", fontsize = 26, rotation =90)
    ax2 = ax1.twinx()
    ax2.plot(x, eigen, color="blue", label=r"$\lambda_{B,1}$")
    ax2.plot(x, eigen_est, color="green", label=r"$\hat{\lambda}_{B,1}$")
    zeros = np.zeros(11)
    ax2.plot(x, zeros, color="gray", linestyle='--')
    ax2.set_ylabel("eigenvalue", fontsize = 26)
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.show()
    
    
def three_plot3(simu, eigen, eigen_est, epislon, num, task):
    names = ["Enron","oregon", "Gnutella"]
    fig, ax1 = plt.subplots()
    # ax1.xticks(fontsize=20)
    # ax1.yticks(fontsize=20)
    names = ["Enron","oregon", "Gnutella"]
    data_path = "C:/Users/xinji/Documents/理论论文/code2/code/models/Simulation/data/"
    with open(data_path+names[num]+task+".pkl", 'wb') as file:
        pickle.dump((simu, eigen, eigen_est), file)
    x = [ep for ep in epislon]
    plt.rcParams.update({'font.size':24})
    ax1.errorbar(x, y = simu[:,0], yerr = simu[:,1:].T, color="red", label="bad_state")
    ax1.set_xlabel(r"$\epsilon$", fontsize = 26)
    ax1.set_ylabel(r"$\bar{x_{-1}}+\bar{x_{-2}} + \bar{x_{-3}}$", fontsize = 26, rotation = 90)
    ax2 = ax1.twinx()
    ax2.plot(x, eigen, color="blue", label=r"$\lambda_{B,1}$")
    ax2.plot(x, eigen_est, color="green", label=r"$\hat{\lambda}_{B,1}$")
    zeros = np.zeros(11)
    ax2.plot(x, zeros, color="gray", linestyle='--')
    ax2.set_ylabel("eigenvalue", fontsize = 26)
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.show()
                 