"""
Store info for joint estimation across loci
"""

import numpy as np
import random
import scipy.integrate
import sys

def ERROR(msg):
    sys.stderr.write(msg.strip() + "\n")
    sys.exit(1)
    
def MSG(msg):
    sys.stderr.write(msg.strip() + "\n")

class JointLocus:
    def __init__(self, _locilist, _ires=10):
        self.loci = _locilist
        self.best_res = None
        self.numiter = 2
        self.method = "Nelder-Mead"
        self.max_cycle_per_iter = 250
        self.ires = _ires

    def callback_function(self, val):
        print("Current parameters: %s"%(str(val)))

    def NegativeLogLikelihood(self, params, numfeatures, drawmu, mu_bounds, sd_bounds, \
                                  beta_bounds, pgeom_bounds, debug=False):
        # Order of params: [mu0, mu_coeff1, mu_coeff2, mu_coeff3...] if drawmu=False
        # Order of params: [mu0, sd0, mu_coeff1, coeff2, ... sd_coeff1, sd_coeff2,...] if drawmu=True
        mu = params[0]
        feature_param_index = 1
        if drawmu:
            sd = params[1]
            feature_param_index = 2
        # Check bounds
        if mu < np.log10(mu_bounds[0]) or mu > np.log10(mu_bounds[1]): return np.inf
        if drawmu:
            if sd < sd_bounds[0] or sd > sd_bounds[1]: return np.inf
        # Loop over each locus
        nloglik = 0
        for i in range(len(self.loci)):
            # Adjust mu for features
            adj_mu = mu + sum([self.loci[i].features[j]*params[j+feature_param_index] for j in range(numfeatures)])
            if drawmu:
                adj_sd = sd + sum([self.loci[i].features[j]*params[j+feature_param_index+numfeatures] for j in range(numfeatures)])
                # Draw samples for mu
                musamples = np.random.normal(loc=adj_mu, scale=adj_sd, size=self.ires)
            else:
                musamples = [adj_mu]
            # Approximate integration of P(D|mu)P(u)du
            nll_values = [self.GetLocusNegLogLikelihood(i, mval, mu_bounds, \
                                                            beta_bounds, pgeom_bounds, debug=debug) for mval in musamples]
            nll = self.GetProbAvg(nll_values)
            if debug: MSG("Locus %s:%s, features: %s, params %s, nll %s"%(self.loci[i].chrom, self.loci[i].start, \
                                                                              str(self.loci[i].features), str(params), nll))
            nloglik += nll
            if nloglik >= np.inf: return nloglik # no point in continuing...
        if debug: MSG("Params %s, Likelihood %s"%(str(params), nloglik))
        return nloglik

    def GetProbAvg(self, nll_values):
        # TODO precision?
        return -1*np.log(np.mean([np.exp(-1*val) for val in nll_values]))

    def GetLocusNegLogLikelihood(self, locindex, mu, mu_bounds, \
                                     beta_bounds, pgeom_bounds, debug=False):
        locus = self.loci[locindex]
        beta = np.random.uniform(*beta_bounds)
        pgeom = np.random.uniform(*pgeom_bounds)
        if locus.prior_beta is not None: beta = locus.prior_beta
        if locus.prior_pgeom is not None: pgeom = locus.prior_pgeom
        val = locus.NegativeLogLikelihood(mu, beta, pgeom, \
                                              mu_bounds, beta_bounds, pgeom_bounds, \
                                              None, None, None, debug=debug)
        return val

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

    def PrintResults(self, out):
        if self.best_res is None: return
        out.write("\t".join(map(str, ["JOINT"]+list(self.best_res.x) + ["%s loci"%len(self.loci)]))+"\n")
        out.flush()
