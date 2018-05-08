import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

os.getcwd() # 기본 디렉토리 확인
os.chdir("C:/Users/mjh/Documents/ZeroYieldCurve")

df = pd.read_excel("interest_hw_data.xlsx", sheetname = 0, header=0, index_col=0)

r_nelson = lambda x,m: x[0]+(x[1]*(1-(np.exp(-m/x[3])))*(x[3]/m))+(x[2]*(((1-(np.exp(-m/x[3]))*(x[3]/m))-np.exp(-m/x[3]))))
d_nelson = lambda x,m: np.exp(-r_nelson(x,m)*m/100)
r_svensson = lambda x,m: x[0]+(x[1]*(1-(np.exp(-m/x[4])))*(x[4]/m))+(x[2]*(((1-(np.exp(-m/x[4]))*(x[4]/m))-np.exp(-m/x[4]))))+(x[3]*(((1-(np.exp(-m/x[5]))*(x[5]/m))-np.exp(-m/x[5]))))
d_svensson = lambda x,m: np.exp(-r_svensson(x,m)*m/100)

def nelson(x):
    obj=0
    for i in range(0,4):#무이표채
        price = df.iloc[i,0]
        m = df.iloc[i,5]
        duration = df.iloc[i,4]
        price_e = d_nelson(x,m)*100
        phi = duration*price
        obj_dis = ((price-price_e)/phi)**2
        obj = obj+obj_dis
    
    for k in range(4,df.index.size): #할인채
        start = df.iloc[k,6]
        price = df.iloc[k,0]
        m = df.iloc[k,5]
        coupon = (df.iloc[k,1]) /2
        duration = df.iloc[k,4]
        phi = duration*price
        price_ee = 0
        for j in range(0,1+int(((m-start)*2))):
            q = start+(0.5*j)
            price_ee = price_ee +(d_nelson(x,q)*coupon)
        price_ee = price_ee+(d_nelson(x,m)*100)
        obj_cou = ((price - price_ee)/phi)**2
        obj = obj+obj_cou
    return obj

x0 = [0.1,0.1,0.1,0.1]
res = minimize(nelson,x0,method='Nelder-Mead', tol=1e-10)
a=res.x

def svensson(x):
    obj=0
    for i in range(0,4):#무이표채
        price = df.iloc[i,0]
        m = df.iloc[i,5]
        duration = df.iloc[i,4]
        price_e = d_svensson(x,m)*100
        phi = duration*price
        obj_dis = ((price-price_e)/phi)**2
        obj = obj+obj_dis
    
    for k in range(4,df.index.size): #할인채
        start = df.iloc[k,6]
        price = df.iloc[k,0]
        m = df.iloc[k,5]
        coupon = (df.iloc[k,1]) /2
        duration = df.iloc[k,4]
        phi = duration*price
        price_ee = 0
        for j in range(0,1+int(((m-start)*2))):
            q = start+(0.5*j)
            price_ee = price_ee +(d_svensson(x,q)*coupon)
        price_ee = price_ee+(100*d_svensson(x,m))
        obj_cou = ((price - price_ee)/phi)**2
        obj = obj+obj_cou
    return obj

x0 = [1,1,1,1,1,1]
res = minimize(svensson,x0,method='Nelder-Mead', tol=1e-10)
b=res.x

c = np.linspace(0.1,30,101)
d= r_nelson(a,c)
e= r_svensson(b,c)
plt.plot(c,d/100, label='NS')
plt.plot(c,e/100, label='SV')
plt.legend(loc='upper left')
plt.plot(df['tillMat'],df['YLD_YTM_MID']/100,'gs')
plt.show()