import json, numpy as np
cases=json.load(open('public_cases.json'))
X=[]
Y=[]
for c in cases:
    d=c['input']['trip_duration_days']
    m=c['input']['miles_traveled']
    r=c['input']['total_receipts_amount']
    y=c['expected_output']
    rpday = r/(d+1e-6)
    rpmile = r/(m+1e-6)
    mpday = m/(d+1e-6)
    features=[1,d,m,r,d**2,m**2,r**2,d*m,d*r,m*r,rpday,rpmile,mpday]
    X.append(features)
    Y.append(y)
X=np.array(X)
Y=np.array(Y)
lam=0.01
beta=np.linalg.solve(X.T.dot(X)+lam*np.eye(X.shape[1]), X.T.dot(Y))
yhat=X.dot(beta)
mae=np.mean(np.abs(yhat-Y))
print('MAE',mae)
print('coeff',beta) 