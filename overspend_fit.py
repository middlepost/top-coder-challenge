import json, numpy as np
cases=json.load(open('public_cases.json'))
X=[];Y=[]
for c in cases:
    d=c['input']['trip_duration_days']; m=c['input']['miles_traveled']; r=c['input']['total_receipts_amount']; y=c['expected_output']
    over=max(0, r-800)
    over_per_day=over/(d+1e-6)
    log_r=np.log1p(r)
    log_m=np.log1p(m)
    features=[1,d,m,r,d*d,m*m,r*r,d*m,d*r,m*r,d**3,d*d*m,d*d*r,d*m*m,d*m*r,d*r*r,m**3,m*m*r,m*r*r,r**3,
              r/(d+1e-6), r/(m+1e-6), m/(d+1e-6), over, over_per_day, log_r, log_m]
    X.append(features)
    Y.append(y)
X=np.array(X); Y=np.array(Y)
lam=10.0
coef=np.linalg.solve(X.T.dot(X)+lam*np.eye(X.shape[1]), X.T.dot(Y))
mae=np.mean(np.abs(X.dot(coef)-Y))
print('MAE', mae)
print('len', len(coef))
np.save('overspend_coef.npy', coef) 