import json, numpy as np, math
cases=json.load(open('public_cases.json'))
X=[];Y=[]
for c in cases:
    d=c['input']['trip_duration_days']; m=c['input']['miles_traveled']; r=c['input']['total_receipts_amount']; y=c['expected_output']
    d2=d**2; d3=d**3
    m2=m**2; m3=m**3
    r2=r**2; r3=r**3
    over=max(0,r-800)
    over2=over**2; over3=over**3
    d_safe=d+1e-6; m_safe=m+1e-6
    rpd=r/d_safe; rpp=r/m_safe; mpd=m/d_safe
    logr=math.log1p(r); logm=math.log1p(m)
    lux=(1 if r>1500 else 0)
    lux2=(1 if rpd>200 else 0)
    # features base 33 + new 4 flags interactions
    feats=[1,d,m,r,d2,m2,r2,d*m,d*r,m*r,d3,m3,r3,d2*m,d2*r,d*m2,d*m*r,d*r2,m2*r,m*r2,
           over,over2,over3,over*d,over*m,over*r,rpd,rpp,mpd,1/d_safe,1/m_safe,logr,logm,
           lux,lux*d,lux*r,lux*over,lux2,lux2*d,lux2*r,lux2*over]
    X.append(feats)
    Y.append(y)
X=np.array(X); Y=np.array(Y)
lam=10.0
coef=np.linalg.solve(X.T.dot(X)+lam*np.eye(X.shape[1]), X.T.dot(Y))
mae=np.mean(np.abs(X.dot(coef)-Y))
print('MAE', mae)
print('nfeat', X.shape[1])
np.save('high_coef.npy', coef) 