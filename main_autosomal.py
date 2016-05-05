#!/usr/bin/env python

"""
Estimate autosomal mutation rates using MUTEA
"""

# TODO
# Choose root node prior
# Choose maximization method to be faster? e.g. max mu first then others
# Read in VCF format
# Correct for stutter

import argparse
import math
import numpy as np
import os
import random
import tabix
import scipy.optimize
import sys

from mutation_model import OUGeomSTRMutationModel
import matrix_optimizer

sys.path.append("/home/mag50/workspace/cteam/mutation_models/")
from ModelEstimatorTMRCA import *

SMALLNUM = 10e-20

def ERROR(msg):
    sys.stderr.write(msg.strip() + "\n")
    sys.exit(1)
    
def MSG(msg):
    sys.stderr.write(msg.strip() + "\n")

def LoadLoci(locfile, datafile):
    loci = []
    with open(locfile, "r") as f:
        for line in f:
            chrom, start, end = line.strip().split()
            chrom = str(chrom)
            start = int(start)
            end = int(end)
            loc = Locus(chrom, start, end, datafile)
            loci.append(loc)
    return loci

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
        ERROR("Unable to find an allele range with leakage < the provided bounds and < the specified maximum")

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

def GenerateOptimizer(mut_model):
    optimizer = matrix_optimizer.MATRIX_OPTIMIZER(mut_model.trans_matrix, mut_model.min_n)
    optimizer.precompute_results()
    return optimizer

def GenerateMutationModel(locilist, mu, beta, pgeom):
    max_tmrca = max(map(lambda x: x.GetMaxTMRCA(), locilist))
    min_str = min(map(lambda x: x.GetMinSTR(), locilist))
    max_str = max(map(lambda x: x.GetMaxSTR(), locilist))
    prev_allele_range = max(map(lambda x: x.prev_allele_range, locilist))
    allele_range = determine_allele_range_from_seed(max_tmrca, 10**mu, beta, pgeom, \
                                                        min_str, max_str, prev_allele_range)
    allele_range = allele_range
    for l in locilist: l.prev_allele_range = allele_range
    mut_model = OUGeomSTRMutationModel(pgeom, 10**mu, beta, allele_range)
    return mut_model, allele_range

class Locus:
    def __init__(self, _chrom, _start, _end, _datafile):
        self.chrom = _chrom
        self.start = _start
        self.end = _end
        self.datafile = _datafile
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
        x = tabix.open(self.datafile)
        try:
            records = list(x.query(self.chrom, self.start, self.end))
        except tabix.TabixError: return
        for r in records:
            chrom, start, end, tmrca, sample, a1, a2, period, asd = r
            tmrca = int(float(tmrca)); a1 = int(a1); a2 = int(a2); period = int(period)
            self.data.append((tmrca, a1/period, a2/period))
            if tmrca > self.maxt: self.maxt = tmrca
            if a1 < self.minstr: self.minstr = a1
            if a2 < self.minstr: self.minstr = a2
            if a1 > self.maxstr: self.maxstr = a1
            if a2 > self.maxstr: self.maxstr = a2
            self.period = period

    def callback_function(self, val):
        print("Current parameters: mu=%f\tbeta=%f\tp=%f"%(val[0], val[1], val[2]))

    def DetermineTotalLogLikelihood(self, allele_range, mut_model, optimizer, debug=False, \
                                        calc_root_prior="bykids"):
        loglik = 0
        all_alleles = range(-allele_range, allele_range+1)
        # Process each sample independently
        for sample in self.data:
            tmrca, a1, a2 = sample
            # Get child likelihoods - TODO use genotype posteriors to get child_likelihoods
            a1_index = all_alleles.index(a1)
            a2_index = all_alleles.index(a2)
            child_indices = [a1_index, a2_index]
            child_likelihoods = []
            for i in range(len(child_indices)):
                cl = np.ones((mut_model.max_n-mut_model.min_n+1))*SMALLNUM
                cl[child_indices[i]] = 1
                child_likelihoods.append(cl)
            # Get root priors
            root_probs = np.zeros((mut_model.max_n-mut_model.min_n+1))
            root_priors = np.ones((mut_model.max_n-mut_model.min_n+1))*SMALLNUM
            if calc_root_prior == "average": # use mean of kids
                root = (a1+a2)/2
                root_index = all_alleles.index(root)
                root_priors[root_index] = 1
            elif calc_root_prior == "bykids": # uniform across alleles close to kids
                buf = 3 # TODO make this smarter
                for a in range(min(child_indices)-buf, max(child_indices)+1+buf):
                    try:
                        root_priors[a] = 1
                    except IndexError: continue
                root_priors = root_priors/np.sum(root_priors)
            else: # Uniform across all alleles
                root_priors = root_priors/np.sum(root_priors)
            # Get root probs
            for i in range(len(child_indices)):
                trans_matrix = np.log(optimizer.get_transition_matrix(tmrca)).transpose()
                trans_matrix += np.log(child_likelihoods[i])
                max_vec = np.max(trans_matrix, axis=1)
                new_probs = np.log(np.sum(np.exp(trans_matrix-max_vec), axis=1)) + max_vec
                root_probs += np.array(new_probs.transpose())[0]
            root_probs += np.log(root_priors)
            max_prob = np.max(root_probs)
            tot_prob = np.log(np.sum(np.exp(root_probs-max_prob))) + max_prob
            loglik += tot_prob
        return loglik

    def NegativeLogLikelihood(self, mu, beta, pgeom, \
                                  mu_bounds, beta_bounds, pgeom_bounds, \
                                  root_prior,
                                  mut_model, allele_range, optimizer, debug=False):
        # Check boundaries
        if mu < math.log10(mu_bounds[0]) or mu > math.log10(mu_bounds[1]): return np.inf
        if beta < beta_bounds[0] or beta > beta_bounds[1]: return np.inf
        if pgeom < pgeom_bounds[0] or pgeom > pgeom_bounds[1]: return np.inf

        # Create mutation model
        if mut_model is None:
            mut_model, allele_range = GenerateMutationModel([self], mu, beta, pgeom)

        # Create optimizer
        if optimizer is None:
            optimizer = GenerateOptimizer(mut_model)

        # Calculate negative likelihood
        ll = self.DetermineTotalLogLikelihood(allele_range+mut_model.max_step, mut_model, optimizer, calc_root_prior=root_prior)
        if ll > 0:
            self.DetermineTotalLogLikelihood(allele_range+mut_model.max_step, mut_model, optimizer, calc_root_prior=root_prior, debug=True)
            ERROR("ERROR: Invalid total LL > 0 for parameters %s %s %s"%(mu, beta, pgeom))
        # debug
        if debug: MSG("%s, %s, %s: loglik %s"%(mu, beta, pgeom, ll))
        return -1*ll

    def MaximizeLikelihood(self, mu_bounds=None, beta_bounds=None, pgeom_bounds=None, \
                               root_prior="bykids", grid=False, debug=False):
        # Load data
        self.LoadData()

        # Likelihood function
        fn = (lambda x: self.NegativeLogLikelihood(x[0], x[1], x[2], \
                                                       mu_bounds, beta_bounds, pgeom_bounds, \
                                                       root_prior,
                                                       None, None, None, debug=debug))
        
        # Optimize likelihood
        if debug: callback = self.callback_function
        else: callback = None
        best_res = None
        if grid:
            mu_step = 0.1
            beta_step = 0.1
            pgeom_step = 0.1
            best_res = scipy.optimize.OptimizeResult()
            best_res.x = None
            best_res.fun = np.inf
            for mu in np.arange(math.log10(mu_bounds[0]), math.log10(mu_bounds[1]), mu_step):
                for beta in np.arange(beta_bounds[0], beta_bounds[1], beta_step):
                    for pgeom in np.arange(pgeom_bounds[0], pgeom_bounds[1], pgeom_step):
                        x0 = [mu, beta, pgeom]
                        ll = fn(x0)
                        if ll < best_res.fun:
                            best_res.x = x0
                            best_res.fun = ll
        else:
            for i in xrange(self.numiter):
                while True:
                    x0 = [random.uniform(math.log10(mu_bounds[0]), math.log10(mu_bounds[1])), \
                              random.uniform(*beta_bounds), random.uniform(*pgeom_bounds)]
                    if not np.isnan(fn(x0)): break
                res = scipy.optimize.minimize(fn, x0, callback=callback, method=self.method, \
                                                  options={'maxiter': self.max_cycle_per_iter, 'xtol': 0.001, 'ftol':0.001})
                if best_res is None or (res.success and res.fun < best_res.fun):
                    best_res = res
        self.best_res = best_res
    
    def GetResults(self):
        if self.best_res is None: return None
        return self.best_res.x, self.best_res.fun

    def PrintResults(self):
        sys.stdout.write("\t".join(map(str, [self.chrom, self.start, self.end]+list(self.best_res.x)))+"\n")
        sys.stdout.flush()

    def __str__(self):
        return "[Locus] %s:%s-%s"%(self.chrom, self.start, self.end)

def main():
    ############################
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--debug", help="Print helpful debug messages", action="store_true")

    # I/O data options - TODO change to VCF input + add stutter correction...
    parser.add_argument("--asdhet", help="ASD-Het file. Must be indexed bed file. See help for columns.", type=str, required=True) 
    parser.add_argument("--loci", help="Bed file with loci to process. First three columns are chrom, start, end", type=str, required=True)
    parser.add_argument("--out", help="Output file (default stdout)", type=str, required=False)

    # Filtering options - TODO

    # Estimation options
    parser.add_argument("--joint", help="Estimate parameters jointly across loci", action="store_true")
    parser.add_argument("--root_prior", help="How to estimate root prior. One of uniform,bykids,average", type=str, default="bykids")
    parser.add_argument("--grid", help="Search over grid rather than using minimization alg", action="store_true")
    parser.add_argument("--min_mu", required=False, type=float, default=0.000001, help="Lower optimization boundary for mu.")
    parser.add_argument("--max_mu", required=False, type=float, default=0.1, help="Upper optimization boundary for mu.")
    parser.add_argument("--min_pgeom", required=False, type=float, default=0.7, help="Lower optimization boundary for pgeom.")
    parser.add_argument("--max_pgeom", required=False, type=float, default=1.0, help="Upper optimization boundary for pgeom.")
    parser.add_argument("--min_beta", required=False, type=float, default=0.0, help="Lower optimization boundary for beta.")
    parser.add_argument("--max_beta", required=False, type=float, default=0.9, help="Upper optimization boundary for beta.")


    args = parser.parse_args()
    ############################
    # Check options
    if not os.path.exists(args.asdhet): ERROR("%s does not exist"%args.asdhet)
    if not os.path.exists(args.loci): ERROR("%s does not exist"%args.loci)
    ############################

    # Load loci to process
    loci = LoadLoci(args.loci, args.asdhet)
    
    # Run estimation
    if args.joint:
        pass # Run estimation together
    else:
        for locus in loci:
            locus.MaximizeLikelihood(mu_bounds=(args.min_mu, args.max_mu), \
                                         beta_bounds=(args.min_beta, args.max_beta), \
                                         pgeom_bounds=(args.min_pgeom, args.max_pgeom), \
                                         root_prior=args.root_prior, \
                                         grid=args.grid, \
                                         debug=args.debug)
            locus.PrintResults()

if __name__ == "__main__":
    main()


