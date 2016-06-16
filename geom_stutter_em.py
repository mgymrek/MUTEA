import collections
import numpy
from scipy.misc  import logsumexp
from scipy.stats import geom
import sys

min_iter        = 10
max_iter        = 1000
min_eff_cov     = 1
CONVERGED       = 0
COVERAGE_LIMIT  = 1
ITERATION_LIMIT = 2
log_one_half = numpy.log(0.5)

def ERROR(msg):
    sys.stderr.write(msg.strip() + "\n")
    sys.exit(1)
    
def MSG(msg):
    sys.stderr.write(msg.strip() + "\n")

def run_EM(sample_read_counts, debug=False, diploid=False):
    valid_read_counts = sample_read_counts.values()
    allele_set        = set(reduce(lambda x,y:x+y, map(lambda x: x.keys(), valid_read_counts)))
    allele_sizes      = sorted(list(allele_set)) # Size of allele for each index
    allele_indices    = dict(map(reversed, enumerate(allele_sizes)))
    eff_coverage      = 0  # Effective number of reads informative for stutter inference
    read_counts       = [] # Array of dictionaries, where key = allele index and count = # such reads for a sample
    max_stutter       = 0
    for i in xrange(len(valid_read_counts)):
        sorted_sizes  = sorted(valid_read_counts[i].keys())
        max_stutter   = max(max_stutter, sorted_sizes[-1]-sorted_sizes[0])
        count_dict    = dict([(allele_indices[x[0]], x[1]) for x in valid_read_counts[i].items()])
        eff_coverage += sum(valid_read_counts[i].values())-1
        read_counts.append(count_dict)        
    num_stutters = 1 + 2*min(5, max_stutter) # Number of stutter options considered [-n, -n+1, ..., 0, ..., n-1, n]

    # Check that we have sufficient reads to perform the inference
    if eff_coverage < min_eff_cov:
        return COVERAGE_LIMIT, eff_coverage, None, None, None, None

    # Initialize parameters
    nalleles         = len(allele_sizes)
    nsamples         = len(read_counts)
    log_gt_priors    = init_log_gt_priors(read_counts, nalleles, diploid=diploid)
    down, up, p_geom = init_stutter_params(read_counts, allele_sizes, diploid=diploid)
    MSG("INIT: P_GEOM=%f, DOWN=%f, UP=%f"%(p_geom, down, up))
    if debug:
        MSG(str(numpy.exp(log_gt_priors)))

    # Construct read count array
    read_counts_array = numpy.zeros((nsamples, nalleles))
    for sample_index,counts in enumerate(read_counts):
        for read_index,count in counts.items():
            read_counts_array[sample_index][read_index] += count

    # Perform EM iterative procedure until convergence
    converged = False
    prev_LL   = -100000000.0
    niter     = 0
    while niter < min_iter or (not converged and niter < max_iter):
        # Recalculate posteriors
        log_gt_posteriors = recalc_log_gt_posteriors(log_gt_priors, down, up, p_geom, read_counts_array, nalleles, allele_sizes, diploid=diploid)
        if debug:
            MSG("POSTERIORS:")
            MSG(str(numpy.exp(log_gt_posteriors)))

        # Reestimate parameters
        log_gt_priors    = recalc_log_pop_priors(log_gt_posteriors)
        down, up, p_geom = recalc_stutter_params(log_gt_posteriors, read_counts, nalleles, allele_sizes, down, up, p_geom, diploid=diploid)
        if debug:
            MSG("POP PRIORS: %s"%(str(numpy.exp(log_gt_priors))))

        if down < 0 or down > 1 or up < 0 or up > 1 or p_geom < 0 or p_geom > 1:
            ERROR("ERROR: Invalid paramter(s) during EM iteration: DOWN=%f, UP=%f, P_GEOM=%f"%(down, up, p_geom))
            
        # Test for convergence
        if niter % 4 == 0:
            new_LL = recalc_log_gt_posteriors(log_gt_priors, down, up, p_geom, read_counts_array, nalleles, allele_sizes, diploid=diploid, norm=True)
            MSG("ITERATION #%d, EM LL=%f, P_GEOM=%f, DOWN=%f, UP=%f"%(niter+1, new_LL, p_geom, down, up))
            converged = (-(new_LL-prev_LL)/prev_LL < 0.0001) and (new_LL-prev_LL < 0.0001)
            prev_LL   = new_LL
        niter    += 1

    if debug: MSG("PRIORS = %s"%(numpy.exp(log_gt_priors)))

    # Return optimized values or placeholders if EM failed
    if niter == max_iter:
        return ITERATION_LIMIT, eff_coverage, down, up, p_geom, new_LL
    else:
        return CONVERGED, eff_coverage, down, up, p_geom, new_LL

def init_log_gt_priors(read_counts, nalleles, diploid=False):
    if diploid:
        gt_counts = numpy.zeros(nalleles**2) + 1.0
    else:
        gt_counts = numpy.zeros(nalleles) + 1.0 # Pseudocount
    for counts in read_counts:
        num_reads = sum(counts.values())
        if diploid:
            gtind = 0
            for a1 in xrange(nalleles):
                for a2 in xrange(nalleles):
                    gt_counts[gtind] += 1.0*counts.get(a1, 0)*counts.get(a2, 0)/num_reads**2 # freq_a1*freq_a2
                    gtind += 1
        else:
            for allele_index,count in counts.items():
                gt_counts[allele_index] += 1.0*count/num_reads
    return numpy.log(1.0*gt_counts/gt_counts.sum())

"""
        if diploid:
            num_gts = len(allele_sizes)**2
            for a1 in xrange(len(allele_sizes)):
                for a2 in xrange(len(allele_sizes)):
                    posterior = counts.get(a1, 0)*counts.get(a2, 0)*1.0/num_reads**2
                    if posterior == 0: continue
                    for read_index, read_count in counts.items():
                        stutter1 = allele_sizes[read_index]-allele_sizes[a1]
                        stutter2 = allele_sizes[read_index]-allele_sizes[a2]
                        print allele_sizes[read_index], allele_sizes[a1], allele_sizes[a2], stutter1, stutter2
                        if abs(stutter1)<abs(stutter2): sval = stutter1
                        else: sval = stutter2
                        dir_counts[numpy.sign(sval)+1] += posterior*read_count
                        diff_sum += read_count*posterior*abs(sval)
        else:
"""
def init_stutter_params(read_counts, allele_sizes, diploid=False):
    if diploid: return 0.01, 0.01, 0.9 # Don't bother trying to get posteriors
    dir_counts = numpy.array([1.0, 1.0, 1.0]) # Pseudocounts -1, 0, 1
    diff_sum = 3.0                            # Step sizes of 1 and 2, so that p_geom < 1 
    for counts in read_counts:
        num_reads = sum(counts.values())
        for allele_index,count in counts.items():
            posterior = 1.0*count/num_reads
            for read_index,read_count in counts.items():
                dir_counts[numpy.sign(read_index-allele_index)+1] += posterior*read_count
                diff_sum += read_count*posterior*abs(allele_sizes[read_index]-allele_sizes[allele_index])
    tot_dir_count = sum(dir_counts)
    return dir_counts[0]/tot_dir_count, dir_counts[2]/tot_dir_count, 1.0*(dir_counts[0]+dir_counts[2])/diff_sum

def recalc_log_pop_priors(log_gt_posteriors):
    log_counts = logsumexp(log_gt_posteriors, axis=0)
    return log_counts - logsumexp(log_counts)

def GetLogTransitionProb(true_allele, obs_allele, down, up, p_geom):
    stutter_dist = geom(p_geom)
    log_down, log_eq, log_up = map(numpy.log, [down, 1-down-up, up])
    if obs_allele - true_allele > 0: logdirprob = log_up
    elif obs_allele - true_allele < 0: logdirprob = log_down
    else: return log_eq
    log_step_prob = stutter_dist.logpmf(abs(obs_allele-true_allele))
    return logdirprob+log_step_prob

def GetReadPhasePosts(allele1, allele2, read_allele, down, up, pgeom):
    log_one_half = numpy.log(0.5)
    log_phase_one = log_one_half + GetLogTransitionProb(allele1, read_allele, down, up, pgeom)
    log_phase_two = log_one_half + GetLogTransitionProb(allele2, read_allele, down, up, pgeom)
    log_phase_total = logsumexp([log_phase_one, log_phase_two])
    return [log_phase_one-log_phase_total, log_phase_two-log_phase_total]

def recalc_stutter_params(log_gt_posteriors, read_counts, nalleles, allele_sizes, down, up, pgeom, diploid=False):
    nsamples   = log_gt_posteriors.shape[0]
    log_counts = [[0], [0], [0]]   # Pseudocounts
    log_diffs  = [0, numpy.log(2)] # Step sizes of 1 and 2, so that p_geom < 1 
    if diploid:
        for i in xrange(nsamples):
            gtind = 0
            for a1 in xrange(nalleles):
                for a2 in xrange(nalleles):
                    log_post = log_gt_posteriors[i][gtind]
                    for read_index, count in read_counts[i].items():
                        log_count = numpy.log(count)
                        diff1 = allele_sizes[read_index]-allele_sizes[a1]
                        diff2 = allele_sizes[read_index]-allele_sizes[a2]
                        phase_posts = GetReadPhasePosts(allele_sizes[a1], allele_sizes[a2], \
                                                            allele_sizes[read_index], down, up, pgeom)
                        diffs = [diff1, diff2]
                        for j in range(len(diffs)):
                            if diffs[j] != 0:
                                log_diffs.append(log_count+log_post+phase_posts[j]+numpy.log(abs(diffs[j])))
                            log_counts[numpy.sign(diffs[j])+1].append(log_post+phase_posts[j]+log_count)
                    gtind += 1
    else:
        for i in xrange(nsamples):
            for j in xrange(nalleles):
                log_post = log_gt_posteriors[i][j]
                for read_index,count in read_counts[i].items():
                    log_count = numpy.log(count)
                    diff      = allele_sizes[read_index] - allele_sizes[j] 
                    if diff != 0:
                        log_diffs.append(log_count + log_post + numpy.log(abs(diff)))
                    log_counts[numpy.sign(diff)+1].append(log_post + log_count)
    log_tot_counts = map(logsumexp, log_counts)
    p_hat          = numpy.exp(logsumexp([log_tot_counts[0], log_tot_counts[2]]) - logsumexp(log_diffs))
    log_freqs      = log_tot_counts - logsumexp(log_tot_counts)
    return numpy.exp(log_freqs[0]), numpy.exp(log_freqs[2]), p_hat

def recalc_log_gt_posteriors(log_gt_priors, down, up, p_geom, read_counts_array, nalleles, allele_sizes, diploid=False, norm=False):
    stutter_dist = geom(p_geom)
    nsamples     = read_counts_array.shape[0]
    log_down, log_eq, log_up = map(numpy.log, [down, 1-down-up, up])
    if diploid:
        num_gts = nalleles**2
        LLs = numpy.zeros((nsamples, num_gts)) + log_gt_priors
        gtind = 0
        for a1 in xrange(nalleles):
            for a2 in xrange(nalleles):
                step_probs1 = numpy.hstack(([log_down + stutter_dist.logpmf(abs(allele_sizes[x]-allele_sizes[a1])) for x in range(0, a1)],
                                            [log_eq],
                                            [log_up   + stutter_dist.logpmf(abs(allele_sizes[x]-allele_sizes[a1])) for x in range(a1+1, nalleles)]))
                step_probs2 = numpy.hstack(([log_down + stutter_dist.logpmf(abs(allele_sizes[x]-allele_sizes[a2])) for x in range(0, a2)],
                                            [log_eq],
                                            [log_up   + stutter_dist.logpmf(abs(allele_sizes[x]-allele_sizes[a2])) for x in range(a2+1, nalleles)]))
                step_probs = numpy.logaddexp(step_probs1, step_probs2)+log_one_half # assume each read equal probability to come from each allele
                LLs[:,gtind] = numpy.sum(read_counts_array*step_probs, axis=1)
                gtind += 1
    else:
        LLs      = numpy.zeros((nsamples, nalleles)) + log_gt_priors
        for j in xrange(nalleles):
            step_probs = numpy.hstack(([log_down + stutter_dist.logpmf(abs(allele_sizes[x]-allele_sizes[j])) for x in range(0, j)],
                                       [log_eq],
                                       [log_up   + stutter_dist.logpmf(abs(allele_sizes[x]-allele_sizes[j])) for x in range(j+1, nalleles)]))
            LLs [:,j] += numpy.sum(read_counts_array*step_probs, axis=1)
    if norm: return numpy.sum(logsumexp(LLs, axis=1))
    else:
        log_samp_totals = logsumexp(LLs, axis=1)[numpy.newaxis].T
        return LLs - log_samp_totals
