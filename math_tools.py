def feynman_point_probability( n, r, b, prob_library={}, max_recursion_depth=20 ):
    """
    Given a base-b number that is n digits long, calculates the probability of a digit occuring r times in succession, using a recursion relation.

    e.g. The probability of '0000000' occuring within the first 16 digits of pi (assuming the digits of pi are 'normal'/randomly sampled).
    3.1415928500000000923429358...

    See github.com/serenasligh/feynman-point-search/math_explanation.pdf for more details.

    n                       - (Int) The number of digits comprising the number.
    r                       - (Int) The number of times a digit occurs successively.
    b                       - (Int) The base of the number.
    prob_library            - (Dict) A dictionary where the output of p(n,r,b) is saved so that it
                              doesn't need to be recalculated.
    max_recursion_depth     - (Int) This prevents recursion problems: Calculation of p(n,r,b)
                              requires p(n-1,r,b) to be calculated, so if n is larger than the
                              maximum recursion depth N in Python, then this function is guaranteed
                              to fail. By saving the output of p(n,r,b) for n < N to prob_library,
                              we can safely calculate p(n,r,b) for arbitrarily large n.
    """

    def p(n,r,b):
        if n<r:
            return 0
        a = 1./b
        if n==r:
            return a**r
        else:
            if (n,r,b) in prob_library.keys():
                return prob_library[(n,r,b)]
            else:
                p_n = p(n-1,r,b) + (1.-p(n-1-r,r,b))*(1.-a)*(a**r)
                prob_library[(n,r,b)] = p_n # Save the output of the calculation to prob_library.
                return p_n

    # We can avoid recursion problems by filling out prob_library.
    i_max = n//max_recursion_depth + 1
    i=1
    while i < i_max:
        p(max_recursion_depth*i,r,b)
        i += 1
    return p(n,r,b)

def gen_primes():
    """
    Sieve of Eratosthenes
    Code by David Eppstein, UC Irvine, 28 Feb 2002
    http://code.activestate.com/recipes/117119/

    Generate an infinite sequence of prime numbers.
    """

    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    D={}
    q=2 # The running integer that's checked for primeness
    while True:
        if q not in D:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            yield q
            D[q*q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # divide it. Since we've reached q, we no longer
            # need it in the map, but we'll mark the next
            # multiples of its witnesses to prepare for larger
            # numbers
            for p in D[q]:
                D.setdefault(p+q,[]).append(p)
            del D[q]
        q += 1

def generate_rationals():
    """
    Generates every positively valued rational number.
    """
    vals = [1,1]
    n = 1
    while True:
        yield (vals[n-1], vals[n]) # (numerator, denominator)
        vals.extend([ vals[n-1]+vals[n], vals[n] ])
        n += 1

    # """
    # Generates every non-negative rational number.
    # """
    # vals = [0,1]
    # n = 1
    # while True:
    #     yield (vals[n-1], vals[n])
    #     vals.append( vals[n] )
    #     n += 1
    #     vals.append( vals[n-1] + vals[n] )

    # vals = (0,1)
    # n = 0
    # while True:
    #     yield (vals[n], vals[n+1])
    #     vals += ( vals[n], vals[n]+vals[n+1] )
    #     n += 1
