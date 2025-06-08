import json, numpy as np, math
cases=json.load(open('public_cases.json'))

def cap_receipt(r):
    if r<=1000: return r
    return 1000+0.3*(r-1000)

def build_row(c):
    d=c['input']['trip_duration_days']
    m=c['input']['miles_traveled']
    r=c['input']['total_receipts_amount']
    rc=cap_receipt(r)
    d2,d3=d**2,d**3
    m2,m3=m**2,m**3
    rc2,rc3=rc**2,rc**3
    r2,r3=r**2,r**3
    over=max(0,r-800)
    over2,over3=over**2,over**3
    d_safe=d+1e-6; m_safe=m+1e-6
    rpd=r/d_safe; rpp=r/m_safe; mpd=m/d_safe
    logr=math.log1p(r); logm=math.log1p(m)
    lux=int(r>1500); lux2=int(rpd>200)
    return [1,d,m,r,rc,d2,m2,rc2,d*m,d*rc,m*rc,d3,m3,rc3,d2*m,d2*rc,d*m2,d*m*rc,d*rc2,m2*rc,m*rc2,
            over,over2,over3,over*d,over*m,over*rc,rpd,rpp,mpd,1/d_safe,1/m_safe,logr,logm,lux,lux*d,lux*rc,lux*over,lux2,lux2*d,lux2*rc,lux2*over]

X=[];Y=[]
for c in cases:
    X.append(build_row(c))
    Y.append(c['expected_output'])
X=np.array(X);Y=np.array(Y)
print('features',X.shape[1])
for lam in [1,2,3,5,8,10]:
    coef=np.linalg.solve(X.T.dot(X)+lam*np.eye(X.shape[1]), X.T.dot(Y))
    mae=float(np.mean(np.abs(X.dot(coef)-Y)))
    print(lam, mae)
np.save('better_X.npy',X)
np.save('better_Y.npy',Y) 