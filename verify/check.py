import json,numpy as np,statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
d=json.load(open('verify/js_out.json'));X=np.array(d['X']);y=np.array(d['y']);y=np.log(y) if d['log'] else y
tr=d['train'];s=d['sel'];A=sm.add_constant(X[tr][:,s]);m=sm.OLS(y[tr],A).fit()
jb=np.array([c['b'] for c in d['coef']]);jp=np.array([c['p'] for c in d['coef']]);jse=np.array([c['se'] for c in d['coef']])
print('max|db|',np.max(np.abs(jb-m.params)/np.maximum(1,np.abs(m.params))),'max|dse| rel',np.max(np.abs(jse-m.bse)/m.bse),'max|dp|',np.max(np.abs(jp-m.pvalues)))
print('F',d['F'],m.fvalue,'R2',d['r2'],m.rsquared)
pr=m.get_prediction(np.r_[1,X[d['x0test']][s]][None]).summary_frame(alpha=.05)
print('PI sm',float(pr.obs_ci_lower.iloc[0]),float(pr.obs_ci_upper.iloc[0]),'js',d['pred']['fit']-d['pred']['pi'],d['pred']['fit']+d['pred']['pi'])
print('CI sm',float(pr.mean_ci_lower.iloc[0]),'js',d['pred']['fit']-d['pred']['ci'])
V=[vif(A,j) for j in range(1,A.shape[1])];print('VIF maxdiff',max(abs(a-b)/b for a,b in zip(V,d['vif'])))
