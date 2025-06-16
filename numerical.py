# 加法模型，分别包括模型生成，以及离散化采样
from scipy.integrate import solve_ivp, odeint
import numpy as np
import networkx as nx
from numba import jit
from operations import add_bad_mtd, cal_largest_eigen, add_bad_active, read_txt_direct, read_txt_undirect, cal_B0_largest_eigen
import random
from collections import Counter
import pickle
# @jit
def fast_ode3(t, x, model_parameters, G):
    # 兼容无向图和有向图
    # 两个好节点两个坏节点
    # 专门写好
    # 默认参数为D
    # x 初值 t是时间
    # 维数为 n,m+n
    # 理解矩阵的形状 
    # 不再用矩阵运算, 用矩阵的度去算
    # x是一个5*n_nodes的向量 顺序依次是 2, 1, -1, -2, -3
    n_nodes = G.number_of_nodes()  # 节点个数
    AT = nx.adjacency_matrix(G).T
    # 提取参数
    p12 = model_parameters["beta"]["p_12"]
    p21 = model_parameters["beta"]["p_21"]
    pm11 = model_parameters["beta"]["p_m11"]
    pm12 = model_parameters["beta"]["p_m12"]
    pm21 = model_parameters["beta"]["p_m21"]
    pm22 = model_parameters["beta"]["p_m22"]
    pm31 = model_parameters["beta"]["p_m31"]
    pm32 = model_parameters["beta"]["p_m32"]
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
    dxdt = np.zeros_like(x)
    Am1= AT@x[2*n_nodes:3*n_nodes]  #-1
    Am2 = AT@x[3*n_nodes:4*n_nodes]  #-2
    Am3 = AT@x[4*n_nodes:5*n_nodes]  #-3
    dxdt[0:n_nodes] = -(p21+(gamma_2m1m1+gamma_2m1m2+gamma_2m1m3)*Am1 + (gamma_2m2m1+gamma_2m2m2+gamma_2m2m3)*Am2)*x[0:n_nodes] + p12*x[n_nodes:2*n_nodes]+pm12*x[2*n_nodes:3*n_nodes] +pm22*x[3*n_nodes:4*n_nodes] + pm32*x[4*n_nodes:5*n_nodes] # 状态2的方程
    dxdt[n_nodes:2*n_nodes] = p21*x[0:n_nodes] -(p12+(gamma_1m1m1+gamma_1m1m2+gamma_1m1m3)*Am1 + (gamma_1m2m1+gamma_1m2m2+gamma_1m2m3)*Am2)*x[n_nodes:2*n_nodes] + pm11*x[2*n_nodes:3*n_nodes] +pm21*x[3*n_nodes:4*n_nodes] + pm31*x[4*n_nodes:5*n_nodes] # 状态1的方程
    dxdt[2*n_nodes:3*n_nodes] = pm2m1*x[3*n_nodes:4*n_nodes] - (pm1m2+pm11+pm12+pm1m3+ gamma_m1m2m2*Am2+gamma_m1m3m3*Am3)*x[2*n_nodes:3*n_nodes] + pm3m1*x[4*n_nodes:5*n_nodes] + (gamma_1m1m1*Am1+gamma_1m2m1*Am2)*x[n_nodes:2*n_nodes] + (gamma_2m1m1*Am1+gamma_2m2m1*Am2)*x[0:n_nodes] +gamma_m2m1m1*Am1*x[3*n_nodes: 4*n_nodes] # 状态-1的方程
    dxdt[3*n_nodes:4*n_nodes] = pm1m2*x[2*n_nodes:3*n_nodes] + pm3m2*x[4*n_nodes:5*n_nodes] - (pm2m3+pm2m1+pm21+pm22+gamma_m2m1m1*Am1+gamma_m2m3m3*Am3)*x[3*n_nodes:4*n_nodes] + (gamma_1m1m2*Am1+gamma_1m2m2*Am2)*x[n_nodes:2*n_nodes] + (gamma_2m1m2*Am1+gamma_2m2m2*Am2)*x[0:n_nodes] +gamma_m1m2m2*Am2*x[2*n_nodes: 3*n_nodes]
    dxdt[4*n_nodes:5*n_nodes] = pm1m3*x[2*n_nodes:3*n_nodes] + pm2m3*x[3*n_nodes:4*n_nodes] - (pm3m1+pm3m2+pm31+pm32)*x[4*n_nodes:5*n_nodes] + (gamma_1m1m3*Am1+gamma_1m2m3*Am2)*x[n_nodes:2*n_nodes] + (gamma_2m1m3*Am1+gamma_2m2m3*Am2)*x[0:n_nodes]+ gamma_m1m3m3*Am3*x[2*n_nodes: 3*n_nodes] + gamma_m2m3m3*Am3*x[3*n_nodes: 4*n_nodes] # 状态-3的方程
    return dxdt

def fast_ode2(t, x, model_parameters, G):
    # 兼容无向图和有向图
    # 两个好节点两个坏节点
    # 专门写好
    # 默认参数为D
    # x 初值 t是时间
    # 维数为 n,m+n
    # 理解矩阵的形状 
    # 不再用矩阵运算, 用矩阵的度去算
    # x是一个5*n_nodes的向量 顺序依次是 2, 1, -1, -2, -3
    n_nodes = G.number_of_nodes()  # 节点个数
    AT = nx.adjacency_matrix(G).T
    # 提取参数
    p12 = model_parameters["beta"]["p_12"]
    p21 = model_parameters["beta"]["p_21"]
    pm11 = model_parameters["beta"]["p_m11"]
    pm12 = model_parameters["beta"]["p_m12"]
    pm21 = model_parameters["beta"]["p_m21"]
    pm22 = model_parameters["beta"]["p_m22"]
    pm1m2 = model_parameters["beta"]["p_m1m2"]
    pm2m1 = model_parameters["beta"]["p_m2m1"]
    gamma_1m1m1 = model_parameters["gamma"]["gamma_1m1m1"]
    gamma_1m1m2 = model_parameters["gamma"]["gamma_1m1m2"]
    gamma_1m2m1 = model_parameters["gamma"]["gamma_1m2m1"]
    gamma_1m2m2 = model_parameters["gamma"]["gamma_1m2m2"]
    gamma_2m1m1 = model_parameters["gamma"]["gamma_2m1m1"]
    gamma_2m1m2 = model_parameters["gamma"]["gamma_2m1m2"]
    gamma_2m2m1 = model_parameters["gamma"]["gamma_2m2m1"]
    gamma_2m2m2 = model_parameters["gamma"]["gamma_2m2m2"]
    gamma_m2m1m1 = model_parameters["gamma"]["gamma_m2m1m1"]
    gamma_m1m2m2 = model_parameters["gamma"]["gamma_m1m2m2"]
    dxdt = np.zeros_like(x)
    Am1= AT@x[2*n_nodes:3*n_nodes]
    Am2 = AT@x[3*n_nodes:4*n_nodes]
    dxdt[0:n_nodes] = -(p21+(gamma_2m1m1+gamma_2m1m2)*Am1 + (gamma_2m2m1+gamma_2m2m2)*Am2)*x[0:n_nodes] + p12*x[n_nodes:2*n_nodes]+pm12*x[2*n_nodes:3*n_nodes] +pm22*x[3*n_nodes:4*n_nodes] # 状态2的方程
    dxdt[n_nodes:2*n_nodes] = p21*x[0:n_nodes] -(p12+(gamma_1m1m1+gamma_1m1m2)*Am1 + (gamma_1m2m1+gamma_1m2m2)*Am2)*x[n_nodes:2*n_nodes]+pm11*x[2*n_nodes:3*n_nodes] +pm21*x[3*n_nodes:4*n_nodes] # 状态1的方程
    dxdt[2*n_nodes:3*n_nodes] = pm2m1*x[3*n_nodes:4*n_nodes] - (pm1m2+pm11+pm12+gamma_m1m2m2*Am2)*x[2*n_nodes:3*n_nodes]  + (gamma_1m1m1*Am1+gamma_1m2m1*Am2)*x[n_nodes:2*n_nodes] + (gamma_2m1m1*Am1+gamma_2m2m1*Am2)*x[0:n_nodes] +gamma_m2m1m1*Am1*x[3*n_nodes: 4*n_nodes]  # 状态-1的方程
    dxdt[3*n_nodes:4*n_nodes] = pm1m2*x[2*n_nodes:3*n_nodes] - (pm2m1+pm21+pm22+gamma_m2m1m1*Am1)*x[3*n_nodes:4*n_nodes] + (gamma_1m1m2*Am1+gamma_1m2m2*Am2)*x[n_nodes:2*n_nodes] + (gamma_2m1m2*Am1+gamma_2m2m2*Am2)*x[0:n_nodes] +gamma_m1m2m2*Am2*x[2*n_nodes: 3*n_nodes]
    return dxdt

def resolve3(G, model_parameters, time):
    rates = model_parameters["infection_rate"]
    n_nodes = G.number_of_nodes()
    y0 = np.zeros(n_nodes*5)
    y0[0:n_nodes] = rates["2_infected"]
    y0[n_nodes:2*n_nodes] = rates["1_infected"]
    y0[2*n_nodes:3*n_nodes] = rates["m1_infected"]
    y0[3*n_nodes:4*n_nodes] = rates["m2_infected"]
    y0[4*n_nodes:5*n_nodes] = rates["m3_infected"]
    t_span = (0, time)
# 评估解的时间点]
    t_eval = np.linspace(0, time, time*10)
    sol = solve_ivp(lambda t, y: fast_ode3(t, y, model_parameters, G), t_span, y0, t_eval=t_eval, method='RK23')
    return sol

def resolve2(G, model_parameters, time):
    rates = model_parameters["infection_rate"]
    n_nodes = G.number_of_nodes()
    y0 = np.zeros(n_nodes*4)
    y0[0:n_nodes] = rates["2_infected"]
    y0[n_nodes:2*n_nodes] = rates["1_infected"]
    y0[2*n_nodes:3*n_nodes] = rates["m1_infected"]
    y0[3*n_nodes:4*n_nodes] = rates["m2_infected"]
    t_span = (0, time)
# 评估解的时间点]
    t_eval = np.linspace(0, time, time*10)
    sol = solve_ivp(lambda t, y: fast_ode2(t, y, model_parameters, G), t_span, y0, t_eval=t_eval, method='RK23')
    return sol

def resolve_ode(G, model_parameters, time):
    rates = model_parameters["infection_rate"]
    n_nodes = G.number_of_nodes()
    y0 = np.zeros(n_nodes*5)
    y0[0:n_nodes] = rates["2_infected"]
    y0[n_nodes:2*n_nodes] = rates["1_infected"]
    y0[2*n_nodes:3*n_nodes] = rates["m1_infected"]
    y0[3*n_nodes:4*n_nodes] = rates["m2_infected"]
    y0[4*n_nodes:5*n_nodes] = rates["m3_infected"]
    # t_span = (0, time)
# 评估解的时间点]
    t_eval = np.linspace(0, time, time*10)
    sol = odeint(lambda y, t: fast_ode3(t, y, model_parameters, G), y0, t_eval )
    return sol

def resolve_ode2(G, model_parameters, time):
    rates = model_parameters["infection_rate"]
    n_nodes = G.number_of_nodes()
    y0 = np.zeros(n_nodes*4)
    y0[0:n_nodes] = rates["2_infected"]
    y0[n_nodes:2*n_nodes] = rates["1_infected"]
    y0[2*n_nodes:3*n_nodes] = rates["m1_infected"]
    y0[3*n_nodes:4*n_nodes] = rates["m2_infected"]
    # t_span = (0, time)
# 评估解的时间点]
    t_eval = np.linspace(0, time, time*10)
    sol = odeint(lambda y, t: fast_ode2(t, y, model_parameters, G), y0, t_eval )
    return sol

def resolve_ode3cas(G, model_parameters, time):
    # 不要精度，串行跑Enron
    rates = model_parameters["infection_rate"]
    n_nodes = G.number_of_nodes()
    y0 = np.zeros(n_nodes*5)
    y0[0:n_nodes] = rates["2_infected"]
    y0[n_nodes:2*n_nodes] = rates["1_infected"]
    y0[2*n_nodes:2*n_nodes] = rates["m1_infected"]
    y0[3*n_nodes:4*n_nodes] = rates["m2_infected"]
    y0[4*n_nodes:5*n_nodes] = rates["m3_infected"]
    t_span = (0, 1)
    t_eval = np.linspace(0, 1, 10)
    sol = solve_ivp(lambda t, y: fast_ode3(t, y, model_parameters, G), t_span, y0, t_eval=t_eval, method='RK23')
    for i in range(1, time//2):
        y0 = sol.y[:,-1]
        t_span = (0, 2)
        t_eval = np.linspace(0, 2, 20)
        sol = solve_ivp(lambda t, y: fast_ode3(t, y, model_parameters, G), t_span, y0, t_eval=t_eval, method='RK23')
    # sol = odeint(lambda y, t: fast_ode3(t, y, model_parameters, G), y0, t_eval )
    return sol


# 计算出结果后抽样的函数
def sample(state_prob, n_nodes, m_state):
    state_prob.reshape(len(state_prob),-1)
    # 要分清顺序
    # cnt = Counter()
    num_state = 2+m_state
    cnt = [0]*num_state
    state_prob = state_prob.reshape((num_state, n_nodes)).T
    state_prob = np.concatenate((np.zeros((n_nodes,1)), state_prob), axis = 1)
    for i in range(1,num_state+1):
        state_prob[:,i] += state_prob[:,i-1]
    seed = np.random.uniform(0,1,n_nodes)
    for i in range(1,num_state+1):
        cnt[i-1] = sum((seed>state_prob[:,i-1])*(seed<=state_prob[:,i]))
    return cnt

def numerical(model_parameters, G, time):
    sol = resolve2(G, model_parameters, time)
    return sample(sol.y[-1], G.number_of_nodes, 2)
    
    
    
    


if __name__ == "__main__":
    data1 = 'C:/Users/xinji/Documents/理论论文/code2/code/networks/Email-Enron.txt' # 无向图
    data2 = 'C:/Users/xinji/Documents/理论论文/code2/code/networks/oregon2_010526.txt' # 无向图
    data3 = 'C:/Users/xinji/Documents/理论论文/code2/code/networks/p2p-Gnutella05.txt' # 有向图
    # 计算矩阵特征值
    # 生成数据
    # 画图
    g1 = read_txt_undirect(data1)
    g2 = read_txt_undirect(data2)
    g3 = read_txt_direct(data3)


    propor = 0.1
    gamma = 0.0005
    p = 0.4
    model_parameters = {"gamma":{
        "gamma_1m1m1": gamma, "gamma_1m1m2": gamma, "gamma_1m1m3":0,
        "gamma_1m2m1": gamma, "gamma_1m2m2": gamma, "gamma_1m2m3":0,
        "gamma_1m3m1": 0, "gamma_1m3m2": 0, "gamma_1m3m3": 0,
        "gamma_2m1m1": gamma, "gamma_2m1m2": gamma, "gamma_2m1m3":0,
        "gamma_2m2m1":gamma, "gamma_2m2m2": gamma, "gamma_2m2m3": 0,
        "gamma_2m3m1": 0, "gamma_2m3m2": 0, "gamma_2m3m3":0,
        "gamma_3m1m1": 0,  "gamma_3m1m2": 0, "gamma_3m1m3": 0, 
        "gamma_3m2m1": 0,  "gamma_3m2m2": 0, "gamma_3m2m3":0,
        "gamma_3m3m1":0, "gamma_3m3m2": 0, "gamma_3m3m3": 0,
        "gamma_m1m2m2": 0.003, "gamma_m1m2m3": 0, 
        "gamma_m1m3m2":0, "gamma_m1m3m3":0,
        "gamma_m2m1m1": 0.003, "gamma_m2m1m3": 0,
        "gamma_m2m3m3": 0, "gamma_m2m3m1": 0,
        "gamma_m3m1m1":0, "gamma_m3m1m2": 0,
        
        "gamma_m3m2m1":0, "gamma_m3m2m2":0},
    "beta":{
                "p_12":0.2, "p_13":0,
                "p_23":0, "p_21": 0.3, "p_23": 0,
                "p_31":0, "p_32": 0, 
                "p_m11":p, "p_m12": p, "p_m13":0,
                "p_m21": p, "p_m22": p, "p_m23": 0, 
                "p_m31": 0, "p_m32":0, "p_m33": 0,
                "p_m1m2": 0, "p_m1m3":0, 
                "p_m2m1": 0, "p_m2m3": 0,
                "p_m3m1": 0, "p_m3m2": 0
    },
    "infection_rate":{"3_infected":0, "2_infected":(1-propor)/2, "1_infected":(1-propor)/2, "m1_infected":propor/2, "m2_infected":propor/2, "m3_infected":0}              
    }    
    # fast_ode3(0, x, model_parameters, g1)
    # sol = resolve(g2, model_parameters, 10)
    # print(sol.y.shape)
    
    