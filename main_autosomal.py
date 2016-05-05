#!/usr/bin/env python

"""
Estimate autosomal mutation rates using MUTEA
"""

# TODO
# Choose maximization method to be faster? e.g. max mu first then others
# Read in VCF format
# Correct for stutter
# Use matrix math for asd distribution

import argparse
import numpy as np
import os
import scipy.optimize
import scipy.stats
import sys
import locus


def LoadLoci(locfile, datafile, minsamples, maxsamples):
    loci = []
    with open(locfile, "r") as f:
        for line in f:
            chrom, start, end = line.strip().split()[0:3]
            chrom = str(chrom)
            start = int(start)
            end = int(end)
            loc = locus.Locus(chrom, start, end, datafile, minsamples, maxsamples)
            loci.append(loc)
    return loci

def main():
    ############################
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--debug", help="Print helpful debug messages", action="store_true")

    # I/O data options
    parser.add_argument("--asdhet", help="ASD-Het file. Must be indexed bed file. See help for columns.", type=str, required=True) 
    parser.add_argument("--loci", help="Bed file with loci to process. First three columns are chrom, start, end", type=str, required=True)
    parser.add_argument("--out", help="Output file (default stdout)", type=str, required=False)

    # Filtering options
    parser.add_argument("--min_samples", help="Don't process if less than this many samples", type=int, default=50)
    parser.add_argument("--max_samples", help="Only process this many samples (for debugging)", type=int, default=1000000)

    # Estimation options
    parser.add_argument("--min_mu", required=False, type=float, default=0.0000001, help="Lower optimization boundary for mu.")
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
    loci = LoadLoci(args.loci, args.asdhet, args.min_samples, args.max_samples)
    
    # Run estimation
    for locus in loci:
        locus.MaximizeLikelihood(mu_bounds=(args.min_mu, args.max_mu), \
                                     beta_bounds=(args.min_beta, args.max_beta), \
                                     pgeom_bounds=(args.min_pgeom, args.max_pgeom), \
                                     debug=args.debug)
        locus.PrintResults()

if __name__ == "__main__":
    main()


