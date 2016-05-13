#!/usr/bin/env python

"""
Estimate autosomal mutation rates using MUTEA
"""

import argparse
import glob
import joblib
import numpy as np
import os
import scipy.optimize
import scipy.stats
import sys

import locus
import jointlocus

# For profiling
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

def LoadLocusFeatures(locusfeatures):
    if locusfeatures is None: return None
    features = {}
    numfeatures = 0
    with open(locusfeatures, "r") as f:
        for line in f:
            items = line.strip().split()
            chrom = items[0]
            start = int(items[1])
            end = int(items[2])
            fvals = map(float, items[3:])
            numfeatures = len(fvals)
            features[(chrom, start, end)] = fvals
    if numfeatures == 0: return features
    # Center each feature
    feature_means = []
    for i in range(numfeatures):
        feature_means.append(np.mean([features[key][i] for key in features]))
    for key in features:
        features[key] = [features[key][i]-feature_means[i] for i in range(numfeatures)]
    sys.stdout.write("# Feature means: %s\n"%",".join(map(str, feature_means)))
    return features

def LoadPriors(locuspriors):
    if locuspriors is None: return None
    priors = {}
    with open(locuspriors, "r") as f:
        for line in f:
            chrom, start, end, logmu, beta, pgeom, numsamples = line.strip().split()
            start = int(start)
            end = int(end)
            logmu = float(logmu)
            beta = float(beta)
            pgeom = float(pgeom)
            priors[(chrom, start, end)] = (logmu, beta, pgeom)
    return priors

def LoadLoci(locfile, datafiles, minsamples, maxsamples, \
                 locuspriors, locusfeatures, stderrs, jackknife_blocksize):
    priors = LoadPriors(locuspriors)
    features = LoadLocusFeatures(locusfeatures)
    loci = []
    with open(locfile, "r") as f:
        for line in f:
            chrom, start, end = line.strip().split()[0:3]
            chrom = str(chrom)
            start = int(start)
            end = int(end)
            loc = locus.Locus(chrom, start, end, datafiles, minsamples, maxsamples, stderrs)
            loc.jkblocksize = jackknife_blocksize
            key = (chrom, start, end)
            if priors is not None and key not in priors: continue
            if features is not None and key not in features: continue
            if priors is not None and key in priors:
                loc.LoadPriors(*priors[key])
            if features is not None and key in features:
                loc.LoadFeatures(features[key])
            loci.append(loc)
    return loci

def ERROR(msg):
    sys.stderr.write(msg.strip() + "\n")
    sys.exit(1)
    
def MSG(msg):
    sys.stderr.write(msg.strip() + "\n")

def RunLocus(locus, args=None):
    locus.MaximizeLikelihood(mu_bounds=(args.min_mu, args.max_mu), \
                                 beta_bounds=(args.min_beta, args.max_beta), \
                                 pgeom_bounds=(args.min_pgeom, args.max_pgeom), \
                                 debug=args.debug)
    # Print intermediate results to stderr so we can track
    outline = locus.GetResultsString()
    sys.stderr.write("PROGRESS: " + outline)
    return outline

def main():
    ############################
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--debug", help="Print helpful debug messages", action="store_true")
    parser.add_argument("--profile", help="Profile code performance", action="store_true")
    parser.add_argument("--maxloci", help="Only process this many loci (for debugging)", default=np.inf, type=int)
    parser.add_argument("--numproc", help="Number of processors", default=1, type=int)

    # I/O data options
    parser.add_argument("--asdhet", help="ASD-Het file. Must be indexed bed file. See help for columns.", type=str, required=True) 
    parser.add_argument("--asdhetdir", help="Is asdhet a directory", action="store_true")
    parser.add_argument("--loci", help="Bed file with loci to process. First three columns are chrom, start, end", type=str, required=True)
    parser.add_argument("--out", help="Output file (default stdout)", type=str, required=False)

    # Filtering options
    parser.add_argument("--min_samples", help="Don't process if less than this many samples", type=int, default=50)
    parser.add_argument("--max_samples", help="Only process this many samples (for debugging)", type=int, default=1000000)

    # Per-locus Estimation options
    parser.add_argument("--min_mu", required=False, type=float, default=0.0000001, help="Lower optimization boundary for mu.")
    parser.add_argument("--max_mu", required=False, type=float, default=0.1, help="Upper optimization boundary for mu.")
    parser.add_argument("--min_pgeom", required=False, type=float, default=0.7, help="Lower optimization boundary for pgeom.")
    parser.add_argument("--max_pgeom", required=False, type=float, default=1.0, help="Upper optimization boundary for pgeom.")
    parser.add_argument("--min_beta", required=False, type=float, default=0.0, help="Lower optimization boundary for beta.")
    parser.add_argument("--max_beta", required=False, type=float, default=0.9, help="Upper optimization boundary for beta.")
    parser.add_argument("--stderrs", required=False, type=str, default="fisher", help="Method to calc stderrs. Options: fisher, jackknife, both.")
    parser.add_argument("--jackknife_blocksize", required=False, type=int, default=10, help="Jackknife block size.")

    # Joint estimation options
    parser.add_argument("--joint", help="Estimate parameters jointly across loci", action="store_true")
    parser.add_argument("--locus_priors", help="Per-locus results to use as priors. Default: draw from uniform", type=str)
    parser.add_argument("--locus_features", help="Tab file of chrom, start, end, feature1, feature2, ...", type=str)
    parser.add_argument("--drawmu", help="Model mu as drawn from a distribution", action="store_true")
    parser.add_argument("--min_sd", required=False, type=float, default=0.001, help="Lower optimization boundary for sd.")
    parser.add_argument("--max_sd", required=False, type=float, default=3, help="Upper optimization boundary for sd.")
    parser.add_argument("--ires", required=False, type=int, default=100, help="Resolution of integration")

    args = parser.parse_args()
    ############################
    # Check options
    if not os.path.exists(args.asdhet): ERROR("%s does not exist"%args.asdhet)
    if not os.path.exists(args.loci): ERROR("%s does not exist"%args.loci)
    if args.numproc < 1: ERROR("%s must be >=1"%args.numproc)
    if args.locus_priors is not None and not os.path.exists(args.locus_priors): ERROR("%s does not exist"%args.locus_priors)
    if args.locus_features is not None and not os.path.exists(args.locus_features): ERROR("%s does not exist"%args.locus_features)
    ############################

    # Get list of asdhet files
    asdhet = []
    if args.asdhetdir:
        asdhet = glob.glob(args.asdhet + "/*.bed.gz")
    else: asdhet = [args.asdhet]

    # Load loci to process
    loci = LoadLoci(args.loci, asdhet, args.min_samples, args.max_samples, \
                        args.locus_priors, args.locus_features, args.stderrs, args.jackknife_blocksize)
    if len(loci) > args.maxloci: loci = loci[:args.maxloci]
    
    # Get output
    if args.out is None: output = sys.stdout
    else: output = open(args.out, "w")


    ####### Profiling #####
    if args.profile:
        with PyCallGraph(output=GraphvizOutput()):
            for locus in loci: RunLocus(locus, args=args)
        sys.exit(1)
    #######################

    # Run estimation
    if args.joint:
        jlocus = jointlocus.JointLocus(loci, _ires=args.ires, _numproc=args.numproc)
        MSG("[main_autosomal.py] Processing joint locus with %s loci..."%(len(jlocus.loci)))
        jlocus.MaximizeLikelihood(mu_bounds=(args.min_mu, args.max_mu), \
                                      sd_bounds=(args.min_sd, args.max_sd), \
                                      beta_bounds=(args.min_beta, args.max_beta), \
                                      pgeom_bounds=(args.min_pgeom, args.max_pgeom), \
                                      drawmu=args.drawmu, \
                                      debug=args.debug)
        jlocus.PrintResults(output)
    else:
        outlines = joblib.Parallel(n_jobs=args.numproc, verbose=50)(joblib.delayed(RunLocus)(locus, args=args) for locus in loci)
        for l in outlines:
            output.write(l)
            output.flush()

if __name__ == "__main__":
    main()


