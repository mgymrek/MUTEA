#!/usr/bin/env python

"""
Calibrate standard errors to ground truth data
"""

import argparse
import numpy as np
import pandas as pd
import scipy.stats
import sys

SMALLNUM=10e-10
def GetLocusCoverage(logmu_est, logmu_est_stderr, truth, low, high, ff):
    """
    What percent of true interval prob mass
    is contained in our interval
    """
    # Get fudge interval
    mult = 1.96
    fudge_interval_low = logmu_est-logmu_est_stderr*ff*mult*abs(logmu_est)
    fudge_interval_high = logmu_est+logmu_est_stderr*ff*mult*abs(logmu_est)
    # Get true distribution. Assume mean is center, and binomial CI's
    low = np.log10(low)
    high = np.log10(high)
    true_mean = np.mean([low, high])
    true_err = (high-low+SMALLNUM)/(2*mult)
    true_dist = scipy.stats.norm(loc=true_mean, scale=true_err)
    # Get PDF contained in fudge interval
    mass = true_dist.cdf(fudge_interval_high)-true_dist.cdf(fudge_interval_low)
    return mass

def GetCICoverage(data, ff):
    """
    Get CI coverage method 1:
    Each locus, what % of mass of PDF overlaps our 95% CI?
    Report mean across loci
    """
    coverages = []
    for i in range(data.shape[0]):
        coverages.append(GetLocusCoverage(data.logmu_est.values[i], data.logmu_est_stderr.values[i], \
                                              data.truth.values[i], \
                                              data.lowbound.values[i], data.upbound.values[i], ff))
    return np.mean(coverages)

def SampleLocusCoverage(logmu_est, logmu_est_stderr, truth, low, high, ff):
    """
    Draw point from truth interval
    Is it in our 95% CI?
    """
    # Get fudge interval
    mult = 1.96
    fudge_interval_low = logmu_est-logmu_est_stderr*ff*mult*abs(logmu_est)
    fudge_interval_high = logmu_est+logmu_est_stderr*ff*mult*abs(logmu_est)
    # Get true distribution. Assume mean is center, and binomial CI's
    low = np.log10(low)
    high = np.log10(high)
    true_mean = np.mean([low, high])
    true_err = (high-low+SMALLNUM)/(2*mult)
    # Draw point from true and see if covered
    point = np.random.normal(loc=true_mean, scale=true_err)
    return int(point >= fudge_interval_low and point <= fudge_interval_high)

def SampleCICoverage(data, ff):
    """
    Get CI coverage method 2:
    Sample point from interval of each locus
    What fraction of those overlap our 95% CI?
    """
    numsamples = 1
    overlap = []
    for j in range(numsamples):
        success = []
        for i in range(data.shape[0]):
            success.append(SampleLocusCoverage(data.logmu_est.values[i], data.logmu_est_stderr.values[i], \
                                                   data.truth.values[i], \
                                                   data.lowbound.values[i], data.upbound.values[i], ff))
        overlap.append(np.mean(success))
    return np.mean(overlap)

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--ests", help="Output from main_autosomal with estimates", type=str, required=True)
    parser.add_argument("--truth", help="File with chrom, start, end, point_estimate, lower_bound, upper_bound (not log)", type=str, required=True)
    args = parser.parse_args()

    # Load estimates and truth
    ests = pd.read_csv(args.ests, sep="\t", names=["chrom","start","end","logmu_est","beta_est","p_est","logmu_est_stderr","nsamp"])
    truth = pd.read_csv(args.truth, sep="\t", names=["chrom","start","end","truth","lowbound","upbound"])
    data = pd.merge(ests, truth, on=["chrom","start","end"])

    # Scale to have same mean
    scale = np.mean(data["truth"])/np.mean(10**data["logmu_est"])
    data["truth"] = data["truth"]/scale
    data["lowbound"] = data["lowbound"]/scale
    data["upbound"] = data["upbound"]/scale
    sys.stderr.write("Scale factor=%s\n"%scale)

    # Remove points with stderr=nan or stderr=0
    data = data[~np.isnan(data["logmu_est_stderr"])&(data["logmu_est_stderr"]!=0)]

    # Loop through fudge factors
    for ff in np.arange(0.1, 20, 0.1):
        # Method 1: use PDF mass overlap
        coverage = GetCICoverage(data, ff)
        # Method 2: sample from truth intervals
        coverage2 = SampleCICoverage(data, ff)
        sys.stdout.write("\t".join(map(str, [ff, coverage, coverage2]))+"\n")
    

if __name__ == "__main__":
    main()

