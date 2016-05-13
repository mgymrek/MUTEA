import numpy as np
import mutation_model

def ERROR(msg):
    sys.stderr.write(msg.strip() + "\n")
    sys.exit(1)
    
def MSG(msg):
    sys.stderr.write(msg.strip() + "\n")

def factor_power_two(val):
    power   = 0
    factors = []
    while val != 0:
        if (val & 1) != 0:
            factors.append(power)
        val   >>= 1
        power  += 1
    return factors

# Class designed to accelerate repeated calcuations of transition matrices
# Calculate only powers of t that we will use, using powers of 2 to
# speed up calculations
class MATRIX_OPTIMIZER:
    def __init__(self, per_gen_matrix, min_n):
        self.per_gen_matrix = per_gen_matrix
        self.N              = per_gen_matrix.shape[0]
        self.min_n          = min_n
        self.memoized_matries = {}
        self.powers_of_two = {} # i->2^i

    def precompute_results(self, tmrcas):
        # Clear
        self.powers_of_two = {}
        self.memoized_matrices = {}
        # Compute powers of two
        max_power2 = int(np.floor(np.log(np.max(tmrcas))/np.log(2)))
        cur_matrix = self.per_gen_matrix
        for i in xrange(0, max_power2+1):
            self.powers_of_two[i] = cur_matrix
            cur_matrix = np.dot(cur_matrix, cur_matrix)
        # Compute for given tmrcas
        for t in set(tmrcas):
            # Which powers of two do we need
            powers = factor_power_two(t)
            res = reduce(lambda x, y: np.dot(x, y), map(lambda x: self.powers_of_two[x], powers))
            self.memoized_matrices[t] = np.clip(res, 1e-10, 1.0) # clip here since we don't reuse

    def get_transition_matrix(self, num_generations):
        if num_generations not in self.memoized_matrices:
            ERROR("Did not memoize %s"%num_generations)
        return self.memoized_matrices[num_generations]

if __name__ == "__main__":
    mut_model = mutation_model.OUGeomSTRMutationModel(0.9, 0.001, 0.3, 5)
    mo = MATRIX_OPTIMIZER(mut_model.trans_matrix, mut_model.min_n)
    mo.precompute_results([4,100,1000])

