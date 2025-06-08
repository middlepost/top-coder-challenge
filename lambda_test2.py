import numpy as np
X=np.load('better_X.npy'); Y=np.load('better_Y.npy')
for lam in [0.1,0.2,0.3,0.5,0.8,1]:
    coef=np.linalg.solve(X.T.dot(X)+lam*np.eye(X.shape[1]), X.T.dot(Y))
    mae=float(np.mean(np.abs(X.dot(coef)-Y)))
    print(lam, mae) 