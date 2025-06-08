import json, numpy as np, math, sys

data=json.load(open('public_cases.json'))
X=[];Y=[]
for case in data:
    d=case['input']['trip_duration_days']; m=case['input']['miles_traveled']; r=case['input']['total_receipts_amount']; y=case['expected_output']
    X.append([1,d,m,r,d*m,d*r,m*r,d**2,m**2,r**2])
    Y.append(y)
X=np.array(X); Y=np.array(Y)
w=np.linalg.lstsq(X,Y,rcond=None)[0]
print(w.tolist())
print('MAE', float(np.mean(np.abs(X.dot(w)-Y)))) 