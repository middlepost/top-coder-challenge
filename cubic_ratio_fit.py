import json, numpy as np, itertools
cases=json.load(open('public_cases.json'))
X=[]; Y=[]
for case in cases:
    d=case['input']['trip_duration_days']
    m=case['input']['miles_traveled']
    r=case['input']['total_receipts_amount']
    y=case['expected_output']
    rpday=r/(d+1e-6)
    rpmile=r/(m+1e-6)
    mpday=m/(d+1e-6)
    features=[1,d,m,r,d*d,m*m,r*r,d*m,d*r,m*r,d**3,d*d*m,d*d*r,d*m*m,d*m*r,d*r*r,m**3,m*m*r,m*r*r,r**3,rpday,rpmile,mpday]
    X.append(features)
    Y.append(y)
X=np.array(X); Y=np.array(Y)
lam=1.0  # stronger ridge
coef=np.linalg.solve(X.T.dot(X)+lam*np.eye(X.shape[1]), X.T.dot(Y))
print('features',X.shape[1])
print('MAE',np.mean(np.abs(X.dot(coef)-Y)))
print(coef) 