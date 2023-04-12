# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 19:18:55 2023

@author: wyx
"""
#%% 读取数据
import pandas as pd
import numpy as np
from gurobipy import *
from pypower.api import case39
# 定义常数
baseMVA = 100
basekV = 500
baseI = baseMVA*1000/basekV
I_max = 1000*1000/baseI**2
V_max = 1.06*1.06 # 节点电压上限
V_min = 0.94*0.94 # 节点电压下限
c = [0.01, 0.3, 0.2] # 发电成本
# 节点数据
Data = case39() # 读取IEEE39节点数据
node = Data['bus'] # 读取节点数据
bus_num = node[:,0].astype('int') # 节点编号
P_i = {i+1:node[:,2][i]/baseMVA for i in bus_num-1} # 节点有功功率
Q_i = {i+1:node[:,3][i]/baseMVA for i in bus_num-1} # 节点有功功率
G_Data = Data['gen'] # 读取发电机节点数据
gen_num = G_Data[:,0].astype('int') # 发电机节点编号
genP_max = dict(zip(gen_num,G_Data[:,8])) # 发电机电压上限
genQ_max = dict(zip(gen_num,G_Data[:,3])) # 发电机电压下限
commen_num = list(set(bus_num.tolist())-set(gen_num.tolist())) # 非发电机节点
# 支路数据
branch = Data['branch'] # 读取支路数据
f = branch[:, 0].astype('int') # 支路起始节点
t = branch[:, 1].astype('int') # 支路末端节点
ij = list(zip(f,t)) # 线路集合
r = branch[:,2]/(basekV**2/baseMVA) # 电阻有名值化为标幺值
x = branch[:,3]/(basekV**2/baseMVA) #  电抗有名值化为标幺值
r_ij = dict(zip(ij,r)) # 将电阻与支路对应
x_ij = dict(zip(ij,x)) # 将电抗与支路对应
upStream = {Node:branch[branch[:,1]==Node][:,0].astype('int') for Node in bus_num} # 所有节点的上游节点
downStream = {Node:branch[branch[:,0]==Node][:,1].astype('int') for Node in bus_num} # 所有节点的下游节点

#%% 建立模型
model = Model('DistFlow')
GP_i = model.addVars(gen_num,lb=-GRB.INFINITY,name='GP_i') # 发电机有功出力
GQ_i = model.addVars(gen_num,lb=-GRB.INFINITY,name='GQ_i') # 发电机无功出力
P_ij = model.addVars(ij, lb=-GRB.INFINITY,name='P_ij') # 线路无功潮流
Q_ij = model.addVars(ij, lb=-GRB.INFINITY,name='Q_ij') # 线路有功潮流
l_ij = model.addVars(ij, lb=-GRB.INFINITY,ub=I_max,name='l_ij') # 线路电流
v_i = model.addVars(bus_num, lb=V_min,ub=V_max,name='v_i') # 节点电压

#%% 功率平衡约束
# 非发电机节点功率平衡
model.addConstrs((0==P_i[i]+quicksum(P_ij[i,j] 
    for j in downStream[i])-quicksum(P_ij[k,i]-r_ij[k,i]*l_ij[k,i] for k in upStream[i]) 
    for i in commen_num),name='NodePBalance')
model.addConstrs((0==Q_i[i]+quicksum(Q_ij[i,j] 
    for j in downStream[i])-quicksum(Q_ij[k,i]-x_ij[k,i]*l_ij[k,i] for k in upStream[i]) 
    for i in commen_num),name='NodeQBalance')

# 发电机节点功率平衡
model.addConstrs((0==-GP_i[i]+P_i[i]+quicksum(P_ij[i,j] 
    for j in downStream[i])-quicksum(P_ij[k,i]-r_ij[k,i]*l_ij[k,i] for k in upStream[i]) 
    for i in gen_num),name='NodeGPBalance')
model.addConstrs((0==-GQ_i[i]+Q_i[i]+quicksum(Q_ij[i,j] 
    for j in downStream[i])-quicksum(Q_ij[k,i]-x_ij[k,i]*l_ij[k,i] for k in upStream[i]) 
    for i in gen_num),name='NodeGQBalance')

model.addConstrs((GP_i[i]<=genP_max[i] for i in gen_num),name='Pmax') # 发电机有功出力上限
model.addConstrs((GP_i[i]>=0 for i in gen_num),name='Pmin') # 发电机有功出力下限
model.addConstrs((GP_i[i]<=genQ_max[i] for i in gen_num),name='Qmax') # 发电机无功出力上限
model.addConstrs((GP_i[i]>=-genQ_max[i] for i in gen_num),name='Qmin') # 发电机无功出力下限

model.addConstrs((v_i[j]==v_i[i]-2*(r_ij[i,j]*P_ij[i,j]+x_ij[i,j]*Q_ij[i,j])+(r_ij[i,j]**2
                  +x_ij[i,j]**2)*l_ij[i,j] for (i,j) in ij),name='voltage')
model.addConstrs((v_i[i]>=V_min for i in bus_num),name='voltageMin')
model.addConstrs((v_i[i]<=V_max for i in bus_num),name='voltageMax')
model.addConstr((v_i[31]==1),name='slackNode')
model.addConstrs((l_ij[i,j]<=I_max for i,j in ij),name='Lijconstrs')

model.addConstrs((l_ij[i,j]*v_i[i]>=(P_ij[i,j]**2+Q_ij[i,j]**2) for i,j in ij),name='SOC')
# #%% 模型求解
# 定义目标函数
obj = quicksum(c[0]*GP_i[i]**2+c[1]*GP_i[i]+c[2] for i in gen_num)
model.setObjective(obj,GRB.MINIMIZE)
model.optimize()
#%% 输出结果
P,v, i,f = {},{},{},{}
P = [P_ij[i].x for i in ij]
v = [v_i[i].x for i in bus_num]
i = [GP_i[i].x for i in gen_num]
f = [GQ_i[i].x for i in gen_num]
dP = pd.DataFrame(P)
dV =pd.DataFrame(v)
di= pd.DataFrame(i)
df= pd.DataFrame(f)
