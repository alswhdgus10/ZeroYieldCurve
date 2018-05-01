import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

rate=pd.read_excel("rate.xlsx",index_col=0)

def ns(cal1):
    return sum((rate['YLD_YTM_MID']-(cal1[0]+cal1[1]*(1-np.exp(-m/cal1[3]))/(m/cal1[3])+cal1[2]*((1-np.exp(-m/cal1[3]))/(m/cal1[3])-np.exp(-m/cal1[3]))))**2)
def sv(cal2):
    return sum((rate['YLD_YTM_MID']-(cal2[0]+cal2[1]*(1-np.exp(-m/cal2[4]))/(m/cal2[4])+cal2[2]*((1-np.exp(-m/cal2[4]))/(m/cal2[4])-np.exp(-m/cal2[4]))+cal2[3]*((1-np.exp(-m/cal2[5]))/(m/cal2[5])-np.exp(-m/cal2[5]))))**2)

cal1=np.array([0.01,0.01,0.01,0.1])#순서대로 베타0,1,2 타우1
cal2=np.array([0.01,0.01,0.01,0.01,0.1,0.1]) #순서대로 베타0,1,2,3 타우1,2

m=rate['MATURITY_ANNUAL']

#Neslon Siegel Model
res1=minimize(ns,cal1,method='nelder-mead')
res2=minimize(sv,cal2,method='nelder-mead')

cal1=res1.x
cal2=res2.x

ns=(cal1[0]+cal1[1]*(1-np.exp(-m/cal1[3]))/(m/cal1[3])+cal1[2]*((1-np.exp(-m/cal1[3]))/(m/cal1[3])-np.exp(-m/cal1[3])))
sv=(cal2[0]+cal2[1]*(1-np.exp(-m/cal2[4]))/(m/cal2[4])+cal2[2]*((1-np.exp(-m/cal2[4]))/(m/cal2[4])-np.exp(-m/cal2[4]))+cal2[3]*((1-np.exp(-m/cal2[5]))/(m/cal2[5])-np.exp(-m/cal2[5])))

plt.plot(m,ns,'s-')
plt.show()

plt.plot(m,sv,'s-')
plt.show()