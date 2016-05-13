"""
Store info for a single locus
"""
import math
import numpy as np
import tabix
import random
import scipy.misc
import scipy.optimize
import scipy.stats
import sys
from numpy.linalg import inv

from mutation_model import OUGeomSTRMutationModel
#import matrix_optimizer
import matrix_optimizer2

sys.path.append("/home/mag50/workspace/cteam/mutation_models/")
from ModelEstimatorTMRCA import *

SMALLNUM = 10e-200

def ERROR(msg):
    sys.stderr.write(msg.strip() + "\n")
    sys.exit(1)
    
def MSG(msg):
    sys.stderr.write(msg.strip() + "\n")

def determine_allele_range_from_seed(max_tmrca, mu, beta, p_geom,
                                     min_obs_allele, max_obs_allele,
                                     seed, max_possible_range=200,
                                     max_leakage=1e-5, debug=False):
    if debug:
        MSG("Determining allele range for MAX_TMRCA=%d, mu=%f, beta=%f, p=%f"%(max_tmrca, mu, beta, p_geom))
        MSG("Min obs=%s, Max obs=%s"%(min_obs_allele, max_obs_allele))
    min_possible_range = max(-min_obs_allele, max_obs_allele)

    # Make sure allele range is large enough by checking for leakage, starting from seed
    allele_range = max(seed, min_possible_range)
    while allele_range < max_possible_range:
        mut_model = OUGeomSTRMutationModel(p_geom, mu, beta, allele_range)
        leakages  = []
        trans_mat = mut_model.trans_matrix**max_tmrca
        for allele in min_obs_allele, max_obs_allele:
            vec       = np.zeros((mut_model.N,1))
            vec[allele-mut_model.min_n] = 1
            prob = np.array(trans_mat.dot(vec).transpose())[0]
            leakages.append(prob[0]+prob[-1])
        if leakages[0] < max_leakage and leakages[1] < max_leakage:
            break
        allele_range += 1
    if allele_range == max_possible_range:
        MSG("Unable to find an allele range with leakage < the provided bounds and < the specified maximum")
        return None

    # Attempt to reduce allele range, in case seed was larger than needed
    while allele_range >= min_possible_range:
        mut_model = OUGeomSTRMutationModel(p_geom, mu, beta, allele_range)
        leakages  = []
        trans_mat = mut_model.trans_matrix**max_tmrca
        for allele in min_obs_allele, max_obs_allele:
            vec       = np.zeros((mut_model.N,1))
            vec[allele-mut_model.min_n] = 1
            prob = np.array(trans_mat.dot(vec).transpose())[0]
            leakages.append(prob[0]+prob[-1])
        if leakages[0] > max_leakage or leakages[1] > max_leakage:
            break
        allele_range -= 1
    allele_range += 1
    return allele_range

def GenerateOptimizer(mut_model, tmrcas):
    optimizer = matrix_optimizer2.MATRIX_OPTIMIZER(mut_model.trans_matrix, mut_model.min_n)
    optimizer.precompute_results(tmrcas)
    return optimizer

def GenerateMutationModel(locilist, mu, beta, pgeom):
    max_tmrca = max(map(lambda x: x.GetMaxTMRCA(), locilist))
    min_str = min(map(lambda x: x.GetMinSTR(), locilist))
    max_str = max(map(lambda x: x.GetMaxSTR(), locilist))
    prev_allele_range = max(map(lambda x: x.prev_allele_range, locilist))
    allele_range = determine_allele_range_from_seed(max_tmrca, 10**mu, beta, pgeom, \
                                                        min_str, max_str, prev_allele_range)
    if allele_range is None: return None, None
    prev_allele_range = allele_range
    for l in locilist: l.prev_allele_range = allele_range
    mut_model = OUGeomSTRMutationModel(pgeom, 10**mu, beta, allele_range)
    return mut_model, allele_range

class Locus:
    def __init__(self, _chrom, _start, _end, _datafiles, \
                     _minsamples, _maxsamples, _stderrs_method):
        self.chrom = _chrom
        self.start = _start
        self.end = _end
        self.datafiles = _datafiles
        self.minsamples = _minsamples
        self.maxsamples = _maxsamples
        self.stderrs_method = _stderrs_method
        self.jkblocksize = 10
        self.data = []
        self.maxt = 0
        self.minstr = np.inf
        self.maxstr = -1*np.inf
        self.period = 0
        self.method = "Nelder-Mead"
        self.numiter = 2
        self.max_cycle_per_iter = 250
        self.prev_allele_range = 1
        self.best_res = None
        self.stderr = []
        self.numsamples = 0
        # Priors
        self.prior_logmu = None
        self.prior_beta = None
        self.prior_pgeom = None
        # Features
        self.features = []
        
    def LoadFeatures(self, _features):
        self.features = _features

    def LoadPriors(self, _logmu, _beta, _pgeom):
        self.prior_logmu = _logmu
        self.prior_beta = _beta
        self.prior_pgeom = _pgeom

    def GetCurveFit(self):
        """ TODO remove, this is for testing """
        asds = []
        tmrcas = []
        for dp in self.data:
            tmrcas.append(dp[0])
            asds.append((dp[1]-dp[2])**2)
        estimator = LineFitEstimator()
        estimator.SetNumBS(0)
        estimator.SetEstEff(True)
        estimator.SetCovar("identity")
        estimator.SetPriors(mu=0.001, beta=0.3, step=0.9)
        estimator.SetParams(strsd=1.1)
        estimator.LoadData(asds=asds, tmrcas=tmrcas)
        estimator.Predict()
        return estimator

    def GetMaxTMRCA(self):
        return self.maxt

    def GetMinSTR(self):
        return self.minstr
    
    def GetMaxSTR(self):
        return self.maxstr

    def LoadData(self):
        self.data = []
        for df in self.datafiles:
            x = tabix.open(df)
            try:
                records = list(x.query(self.chrom, self.start, self.end))
            except tabix.TabixError: continue
            for r in records:
                chrom, start, end, tmrca, sample, a1, a2, period, asd = r
                tmrca = int(float(tmrca)); a1 = int(a1); a2 = int(a2); period = int(period)
                self.data.append((tmrca, a1/period, a2/period, (a2-a1)**2/period**2))
                if tmrca > self.maxt: self.maxt = tmrca
                if a1 < self.minstr: self.minstr = a1
                if a2 < self.minstr: self.minstr = a2
                if a1 > self.maxstr: self.maxstr = a1
                if a2 > self.maxstr: self.maxstr = a2
                self.period = period
        if len(self.data) > self.maxsamples: self.data = self.data[:self.maxsamples]
        self.numsamples = len(self.data)

    def callback_function(self, val):
        print("Current parameters: mu=%f\tbeta=%f\tp=%f"%(val[0], val[1], val[2]))

    def DetermineSampleASDLogLikelihood(self, sample_index, allele_range, mut_model, optimizer, \
                                            debug=False):
        # TODO use gt posteriors to weight over possible ASDs
        tmrca, a1, a2, asd = self.data[sample_index]
        afreqs = optimizer.get_afreqs(tmrca)
        # Get prob
        shift = int(np.sqrt(asd))
        prob = float(sum([afreqs[i]*afreqs[i+shift] for i in range(len(afreqs)-shift)]))+SMALLNUM
        return np.log(prob)

    def DetermineTotalLogLikelihood(self, allele_range, mut_model, optimizer, keep, debug=False):
        loglik = 0
        # Process each sample independently
        for i in keep:
            loglik += self.DetermineSampleASDLogLikelihood(i, allele_range, mut_model, optimizer, debug=debug)
        return loglik

    def NegativeLogLikelihood(self, mu, beta, pgeom, keep, \
                                  mu_bounds=None, beta_bounds=None, pgeom_bounds=None, \
                                  mut_model=None, allele_range=None, optimizer=None, debug=False):
        # Check boundaries
        if mu_bounds is not None:
            if mu < math.log10(mu_bounds[0]) or mu > math.log10(mu_bounds[1]): return np.inf
        if beta_bounds is not None:
            if beta < beta_bounds[0] or beta > beta_bounds[1]: return np.inf
        if pgeom_bounds is not None:
            if pgeom < pgeom_bounds[0] or pgeom > pgeom_bounds[1]: return np.inf

        # Create mutation model
        if mut_model is None:
            mut_model, allele_range = GenerateMutationModel([self], mu, beta, pgeom)
            if mut_model is None: return np.inf

        # Create optimizer
        if optimizer is None:
            optimizer = GenerateOptimizer(mut_model, [item[0] for item in self.data])

        # Calculate negative likelihood
        ll = self.DetermineTotalLogLikelihood(allele_range+mut_model.max_step, mut_model, optimizer, keep)
        return -1*ll

    def MaximizeLikelihood(self, mu_bounds=None, beta_bounds=None, pgeom_bounds=None, \
                               debug=False, jackknife=False, jkind=[]):
        if not jackknife:
            self.LoadData()
            if len(self.data) < self.minsamples: return
            keep = range(len(self.data))
        else: keep = jkind

        # Likelihood function
        fn = (lambda x: self.NegativeLogLikelihood(x[0], x[1], x[2], keep, \
                                                       mu_bounds=mu_bounds, beta_bounds=beta_bounds, pgeom_bounds=pgeom_bounds, \
                                                       mut_model=None, allele_range=None, optimizer=None, debug=debug))
        
        # Optimize likelihood
        if debug: callback = self.callback_function
        else: callback = None
        best_res = None
        for i in xrange(self.numiter):
            while True:
                x0 = [random.uniform(math.log10(mu_bounds[0]), math.log10(mu_bounds[1])), \
                          random.uniform(*beta_bounds), random.uniform(*pgeom_bounds)]
                if not np.isnan(fn(x0)): break
            res = scipy.optimize.minimize(fn, x0, callback=callback, method=self.method, \
                                              options={'maxiter': self.max_cycle_per_iter, 'xtol': 0.001, 'ftol':0.001})
            if best_res is None or (res.success and res.fun < best_res.fun):
                best_res = res
        if jackknife:
            return best_res.x
        else:
            self.best_res = best_res
            self.CalculateStdErrors(mu_bounds=mu_bounds, beta_bounds=beta_bounds, pgeom_bounds=pgeom_bounds)

    def PartialDerivative(self, func, var=0, n=1, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(args)
        return scipy.misc.derivative(wraps, point[var], n=n, dx=1e-3)

    def GetLogLikelihoodSecondDeriv(self, mu_bounds=None, beta_bounds=None, pgeom_bounds=None):
        # 2nd derivative wrt mu
        deriv = self.PartialDerivative(lambda x: -1*self.NegativeLogLikelihood(x[0], x[1], x[2], range(len(self.data)), \
                                                                                  mu_bounds=mu_bounds, \
                                                                                  beta_bounds=beta_bounds, \
                                                                                  pgeom_bounds=pgeom_bounds), \
                                          var=0, n=2, point=self.best_res.x)
        return deriv
    
    def GetFisherInfo(self, mu_bounds=None, beta_bounds=None, pgeom_bounds=None):
        if self.best_res is None: return
        fisher_info = -1*self.GetLogLikelihoodSecondDeriv(mu_bounds=mu_bounds, \
                                                              beta_bounds=beta_bounds, \
                                                              pgeom_bounds=pgeom_bounds)
        return fisher_info

    def GetJackknifeStderr(self, mu_bounds=None, beta_bounds=None, pgeom_bounds=None):
        numblocks = len(self.data)/self.jkblocksize
        ests = []
        for i in range(numblocks):
            remove = range(i*self.jkblocksize, (i+1)*self.jkblocksize)
            keepind = [j for j in range(len(self.data)) if j not in remove]
            ests.append(self.MaximizeLikelihood(mu_bounds=mu_bounds, beta_bounds=beta_bounds, pgeom_bounds=pgeom_bounds, \
                                                    jackknife=True, jkind=keepind))
        ests = [item[0] for item in ests]
        return np.sqrt((numblocks-1.0)/numblocks * sum([(item-np.mean(ests))**2 for item in ests]))

    def CalculateStdErrors(self, mu_bounds=None, beta_bounds=None, pgeom_bounds=None):
        if self.best_res is None: return
        if self.stderrs_method == "fisher" or self.stderrs_method == "both":
            fisher_info = self.GetFisherInfo(mu_bounds=mu_bounds, beta_bounds=beta_bounds, pgeom_bounds=pgeom_bounds)
            self.stderr.append(np.sqrt(1/fisher_info))
        if self.stderrs_method == "jackknife" or self.stderrs_method == "both":
            self.stderr.append(self.GetJackknifeStderr(mu_bounds=mu_bounds, beta_bounds=beta_bounds, pgeom_bounds=pgeom_bounds))
        if self.stderrs_method not in ["fisher","jackknife","both"]:
            ERROR("Invalid stderr method %s"%self.stderrs_method)

    def GetResults(self):
        if self.best_res is None: return None
        return self.best_res.x, self.best_res.fun

    def PrintResults(self, outfile):
        if self.best_res is None: return
        outfile.write(self.GetResultsString())
        outfile.flush()

    def GetResultsString(self):
        if self.best_res is None: return
        return "\t".join(map(str, [self.chrom, self.start, self.end]+list(self.best_res.x) + self.stderr + [self.numsamples]))+"\n"


    def __str__(self):
        return "[Locus] %s:%s-%s"%(self.chrom, self.start, self.end)

