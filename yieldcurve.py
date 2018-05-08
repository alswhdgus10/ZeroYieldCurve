# -*- coding: utf-8 -*-
"""
Created on Thu May  3 22:53:35 2018

@author: 우람
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 17:52:43 2018

@author: 삼성컴퓨터
"""

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import scipy as sp
import os
import datetime as dt
os.chdir('C:\\Users\우람\Desktop\interest')
df=pd.read_excel('interest_hw_data_r.xlsx')
#%%


x0=[0.03,-0.01,-0.01,0.7] #b0, b1, b2, t1

def NSS(list):
    nss=[]
    for i in np.arange(len(df)):
        x=list[0]+ (list[1]+list[2])*(list[3]/df['MATURITY'][i])*(1-np.exp(-df['MATURITY'][i]/list[3]))-list[2]*np.exp(-df['MATURITY'][i]/list[3])
        nss.append(x)
    nss=pd.DataFrame(nss)
    nss.columns=['nss'] 
    
    pi=pd.Series([[]])
    estprice=pd.Series([[]])
    pmt=pd.Series([[]])
    for i in np.arange(len(nss)):
        a=pd.Series([[]])
        for j in np.arange(round(df['MATURITY'][i])/0.5):
            a[j]=np.exp(-(df['MATURITY'][i]-0.5*j)*nss.iloc[i,0])*(df['CPN'][i]/2)
        pmt[i] = a.sum() if a.sum() else 0
        estprice[i]=pmt[i]+np.exp(-nss.iloc[i,0]*df['MATURITY'][i])*100   
    pi=df['DUR_ADJ_MID']*df['PX_LAST']/(1+df['YLD_YTM_MID']/100)
    diff=np.power((df['PX_LAST']-estprice)/pi,2)
    return diff.sum()

a=sp.optimize.minimize(NSS,x0,method='Nelder-Mead')
p=a['x']
dff=pd.DataFrame(np.linspace(0,30,101), columns=['MATURITY'])
NS=[]
for i in np.arange(0,len(dff)):
    x=p[0]+ (p[1]+p[2])*(p[3]/dff['MATURITY'][i])*(1-np.exp(-dff['MATURITY'][i]/p[3]))-p[2]*np.exp(-dff['MATURITY'][i]/p[3])
    NS.append(x)
NS=pd.DataFrame(NS)

x1=[1,1,1,1,1,1] #b0, b1, b2, b3, t1, t2

def SV(list):
    svs=[]
    for i in np.arange(len(df)):
        x=list[0]+list[1]*(1-np.exp(-df['MATURITY'][i]/list[4]))*(-list[4]/df['MATURITY'][i])+list[2]*((1-np.exp(-df['MATURITY'][i]/list[4]))*(list[4]/df['MATURITY'][i])-np.exp(-df['MATURITY'][i]/list[4]))+list[3]*((1-np.exp(-df['MATURITY'][i]/list[5]))*(list[5]/df['MATURITY'][i])-np.exp(-df['MATURITY'][i]/list[5]))
        svs.append(x)
    svs=pd.DataFrame(svs)
    svs.columns=['svs'] 
    
    svs_pi=pd.Series([[]])
    svs_estprice=pd.Series([[]])
    pmt=pd.Series([[]])
    for i in np.arange(len(svs)):
        a=pd.Series([[]])
        for j in np.arange(round(df['MATURITY'][i])/0.5):
            a[j]=np.exp(-(df['MATURITY'][i]-0.5*j)*svs.iloc[i,0])*(df['CPN'][i]/2)
        pmt[i] = a.sum() if a.sum() else 0
        svs_estprice[i]=pmt[i]+np.exp(-svs.iloc[i,0]*df['MATURITY'][i])*100
    svs_pi=df['DUR_ADJ_MID']*df['PX_LAST']/(1+df['YLD_YTM_MID']/100)
    svs_diff=np.power((df['PX_LAST']-svs_estprice)/svs_pi,2)
    return svs_diff.sum()

b=sp.optimize.minimize(SV,x1,method='Nelder-Mead')
q=b['x']
SVS=[]
for i in np.arange(0,len(dff)):
    x=q[0]+q[1]*(1-np.exp(-dff['MATURITY'][i]/q[4]))*(-q[4]/dff['MATURITY'][i])+q[2]*((1-np.exp(-dff['MATURITY'][i]/q[4]))*(q[4]/dff['MATURITY'][i])-np.exp(-dff['MATURITY'][i]/q[4]))+q[3]*((1-np.exp(-dff['MATURITY'][i]/q[5]))*(q[5]/dff['MATURITY'][i])-np.exp(-dff['MATURITY'][i]/q[5]))
    SVS.append(x)
SVS=pd.DataFrame(SVS)
plt.plot(dff['MATURITY'],SVS, label='Svensson')
plt.plot(dff['MATURITY'],NS, label='Nelson-Siegel')
plt.legend(loc='upper left')
plt.plot(df['MATURITY'],df['YLD_YTM_MID']/100,"gs")
plt.savefig('a.png', dpi=300)
plt.show()
