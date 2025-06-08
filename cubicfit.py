import json, itertools, numpy as np, math, sys

cases=json.load(open('public_cases.json'))
variables=['d','m','r']

# generate feature names combinations up to degree 3
combos=[]
for degree in [1,2,3]:
    for combo in itertools.combinations_with_replacement(variables, degree):
        combos.append(combo)
# ensure 1 constant term
feature_names=['1']
feature_names.extend(['*'.join(c) for c in combos])

X=[]
Y=[]
for case in cases:
    d=case['input']['trip_duration_days']
    m=case['input']['miles_traveled']
    r=case['input']['total_receipts_amount']
    y=case['expected_output']
    val_dict={'d':d,'m':m,'r':r}
    row=[1]
    for combo in combos:
        prod=1
        for var in combo:
            prod*=val_dict[var]
        row.append(prod)
    X.append(row)
    Y.append(y)
X=np.array(X)
Y=np.array(Y)

# solve least squares with ridge regularization small lam=0.01 to avoid overfit
lam=0.01
XtX=X.T.dot(X)
XtX+=lam*np.eye(XtX.shape[0])
XtY=X.T.dot(Y)
coef=np.linalg.solve(XtX,XtY)
print('Features', len(feature_names))
print('MAE', np.mean(np.abs(X.dot(coef)-Y)))
print('Coeffs:')
for name,c in zip(feature_names,coef):
    print(name,c) 