import numpy as np
import pandas as pd
df = pd.read_csv('test.csv')

def linear_regression(df):
    x = df['X'].to_numpy()
    y = df['t'].to_numpy()
    print(x)
    print(y)
    x_mean = np.mean(x)  
    y_mean = np.mean(y)  

    Sxy = np.sum((x - x_mean)*(y - y_mean))#n倍している
    Sxx = np.sum((x - x_mean)**2)
   
    coef = Sxy / Sxx
    intercept = y_mean - coef * x_mean
    return [coef, intercept]
coef_ans, intercept_ans = linear_regression(df)
print(f'{coef_ans} {intercept_ans}')