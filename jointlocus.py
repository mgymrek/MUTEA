"""
Store info for joint estimation across loci
"""

import joblib
import numpy as np
import random
import scipy.integrate
import sys
from numpy.linalg import inv

def ERROR(msg):
    sys.stderr.write(msg.strip() + "\n")
    sys.exit(1)
    
def MSG(msg):
    sys.stderr.write(msg.strip() + "\n")

def GetLocusNegLL(locus, drawmu, params, feature_param_index, numfeatures, mu, sd, \
                      mu_bounds, beta_bounds, pgeom_bounds, ires=100, debug=False):
    # Adjust mu for features
    adj_mu = mu + sum([locus.features[j]*params[j+feature_param_index] for j in range(numfeatures)])
    if drawmu:
        adj_sd = sd + sum([locus.features[j]*params[j+feature_param_index+numfeatures] for j in range(numfeatures)])
        # Draw samples for mu
        musamples = np.random.normal(loc=adj_mu, scale=adj_sd, size=ires)
    else:
        musamples = [adj_mu]
    # Approximate integration of P(D|mu)P(u)du
    nll_values = [GetLocusNegLogLikelihood(locus, mval, mu_bounds, \
                                               beta_bounds, pgeom_bounds, debug=debug) for mval in musamples]
    if -1*np.inf in nll_values: return -1*np.inf
    nll = GetProbAvg(nll_values)
    return nll

def GetLocusNegLogLikelihood(locus, mu, mu_bounds, \
                                 beta_bounds, pgeom_bounds, debug=False):
    beta = np.random.uniform(*beta_bounds)
    pgeom = np.random.uniform(*pgeom_bounds)
    if locus.prior_beta is not None: beta = locus.prior_beta
    if locus.prior_pgeom is not None: pgeom = locus.prior_pgeom
    val = locus.NegativeLogLikelihood(mu, beta, pgeom, range(len(locus.data)), \
                                          mu_bounds=mu_bounds, beta_bounds=beta_bounds, pgeom_bounds=pgeom_bounds, \
                                          mut_model=None, allele_range=None, optimizer=None, debug=debug)
    return val

def GetProbAvg(nll_values):
    # TODO precision?
    return -1*np.log(np.mean([np.exp(-1*val) for val in nll_values]))

class JointLocus:
    def __init__(self, _locilist, _ires=10, _numproc=1):
        self.loci = _locilist
        self.best_res = None
        self.numiter = 3
        self.method = "Nelder-Mead"
        self.max_cycle_per_iter = 250
        self.ires = _ires
        self.stderrs = []
        self.numproc = _numproc

    def callback_function(self, val):
        print("Current parameters: %s"%(str(val)))

    def NegativeLogLikelihood(self, params, numfeatures, drawmu, mu_bounds, sd_bounds, \
                                  beta_bounds, pgeom_bounds, debug=False):
        # Order of params: [mu0, mu_coeff1, mu_coeff2, mu_coeff3...] if drawmu=False
        # Order of params: [mu0, sd0, mu_coeff1, coeff2, ... sd_coeff1, sd_coeff2,...] if drawmu=True
        mu = params[0]
        sd = None
        feature_param_index = 1
        if drawmu:
            sd = params[1]
            feature_param_index = 2
        # Check bounds
        if mu < np.log10(mu_bounds[0]) or mu > np.log10(mu_bounds[1]): return np.inf
        if drawmu:
            if sd < sd_bounds[0] or sd > sd_bounds[1]: return np.inf
        # Loop over each locus
        locnlogliks = joblib.Parallel(n_jobs=self.numproc, batch_size=10)(joblib.delayed(GetLocusNegLL)(self.loci[i], drawmu, params, feature_param_index, numfeatures, mu, sd, \
                                         mu_bounds, beta_bounds, pgeom_bounds, ires=self.ires, debug=debug) for \
                           i in range(len(self.loci)))
        return sum(locnlogliks)

    def LoadData(self):
        toremove = []
        for locus in self.loci:
            locus.LoadData()
            if len(locus.data) <= locus.minsamples:
                toremove.append(locus)
        for locus in toremove: self.loci.remove(locus)

    def MaximizeLikelihood(self, mu_bounds=None, sd_bounds=None, beta_bounds=None, \
                               pgeom_bounds=None, drawmu=False, debug=False):
        if len(self.loci) == 0: return None
        # Load data for each locus
        self.LoadData()

        # How many params? mu + numfeatures
        numfeatures = len(self.loci[0].features)

        # Likelihood function
        fn = (lambda x: self.NegativeLogLikelihood(x, numfeatures, drawmu, mu_bounds, sd_bounds, \
                                                       beta_bounds, pgeom_bounds, debug=debug))
        # Optimize likelihood
        if debug: callback = self.callback_function
        else: callback = None
        best_res = None
        for i in xrange(self.numiter):
            while True:
                x0 = [random.uniform(np.log10(mu_bounds[0]), np.log10(mu_bounds[1]))]
                if drawmu: x0.append(random.uniform(*sd_bounds))
                x0.extend([0 for i in range(numfeatures)]) # start coeff features at 0
                if drawmu: x0.extend([0 for i in range(numfeatures)]) 
                if not np.isnan(fn(x0)): break
            res = scipy.optimize.minimize(fn, x0, callback=callback, method=self.method, \
                                              options={'maxiter': self.max_cycle_per_iter, 'xtol': 0.001, 'ftol':0.001})
            if best_res is None or (res.success and res.fun < best_res.fun):
                best_res = res
        self.best_res = best_res

        # Calculate stderr
        self.CalculateStdErrors(drawmu=drawmu, mu_bounds=mu_bounds, beta_bounds=beta_bounds, pgeom_bounds=pgeom_bounds, sd_bounds=sd_bounds)

    def PartialDerivative(self, func, var=0, n=1, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(args)
        return scipy.misc.derivative(wraps, point[var], n=n, dx=1e-2)

    def GetLogLikelihoodSecondDeriv(self, dim1, dim2, numfeatures, drawmu, \
                                        mu_bounds=None, beta_bounds=None, pgeom_bounds=None, sd_bounds=None):
        deriv1_fnc = (lambda y: self.PartialDerivative(lambda x: -1*self.NegativeLogLikelihood(x, numfeatures, drawmu, \
                                                                                                   mu_bounds, \
                                                                                                   sd_bounds, \
                                                                                                   beta_bounds, \
                                                                                                   pgeom_bounds), \
                                                           var=dim1, n=1, point=y))
        deriv2 = self.PartialDerivative(deriv1_fnc, var=dim2, n=1, point=self.best_res.x)
        return deriv2

    def GetFisherInfo(self, drawmu=False, mu_bounds=None, beta_bounds=None, pgeom_bounds=None, sd_bounds=None):
        if self.best_res is None: return
        numfeatures = len(self.loci[0].features)
        numparams = (1+numfeatures)*(1+drawmu)
        fisher_info = np.zeros((numparams, numparams))
        for i in range(numparams):
            for j in range(numparams):
                fisher_info[i,j] = -1*self.GetLogLikelihoodSecondDeriv(i, j, numfeatures, drawmu, \
                                                                           mu_bounds=mu_bounds, beta_bounds=beta_bounds, pgeom_bounds=pgeom_bounds, \
                                                                           sd_bounds=sd_bounds)
        return fisher_info

    def CalculateStdErrors(self, drawmu=False, mu_bounds=None, beta_bounds=None, pgeom_bounds=None, sd_bounds=None):
        fisher_info = self.GetFisherInfo(drawmu=drawmu, mu_bounds=mu_bounds, beta_bounds=beta_bounds, pgeom_bounds=pgeom_bounds, sd_bounds=sd_bounds)
        self.stderrs = list(np.sqrt(np.diagonal(inv(fisher_info))))

    def PrintResults(self, out):
        if self.best_res is None: return
        out.write("\t".join(map(str, ["JOINT"]+list(self.best_res.x) + self.stderrs + ["%s loci"%len(self.loci)]))+"\n")
        out.flush()
