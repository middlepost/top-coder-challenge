import json, math, statistics, collections
import numpy as np
from itertools import combinations_with_replacement

data = json.load(open('public_cases.json'))
results = []
for case in data:
    d = case['input']['trip_duration_days']
    m = case['input']['miles_traveled']
    r = case['input']['total_receipts_amount']
    out = case['expected_output']
    results.append((d,m,r,out))

# Compute per-day baseline assuming base per diem of 100
per_diem = 100
mileage_rates = []
for d,m,r,out in results:
    base = d*per_diem
    mileage_est = (out-base)
    if m>0:
        mileage_rates.append(mileage_est/m)
print('Mileage rate stats:', statistics.mean(mileage_rates), statistics.median(mileage_rates), min(mileage_rates), max(mileage_rates))

statistics_rates = mileage_rates  # rename
print('Sample count', len(results))
print('Mean mileage rate', statistics.mean(mileage_rates))
print('Median mileage rate', statistics.median(mileage_rates))
print('10th percentile', sorted(mileage_rates)[int(0.1*len(mileage_rates))])
print('90th percentile', sorted(mileage_rates)[int(0.9*len(mileage_rates))])

# compute correlation
X=np.array([[1,d,m,r] for d,m,r,_ in results])
Y=np.array([out for _,_,_,out in results])
print('Correlation days:', np.corrcoef(X[:,1],Y)[0,1])
print('Correlation miles:', np.corrcoef(X[:,2],Y)[0,1])
print('Correlation receipts:', np.corrcoef(X[:,3],Y)[0,1])

# compute linear regression
w=np.linalg.lstsq(X,Y,rcond=None)[0]
print('OLS weights', w)
# compute predictions
pred=X.dot(w)
mae=np.mean(np.abs(pred-Y))
print('MAE', mae)

features=[]
for d,m,r,_ in results:
    feats=[1,d,m,r,d*m,d*r,m*r,d**2,m**2,r**2]
    features.append(feats)
X2=np.array(features)
w2=np.linalg.lstsq(X2,Y,rcond=None)[0]
print('Poly weights length', len(w2))
pred2=X2.dot(w2)
mae2=np.mean(np.abs(pred2-Y))
print('Poly MAE', mae2) 