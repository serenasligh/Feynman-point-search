def standard_notation(number_str):
    """
    Converts a number, repsented as a string in any base, from scientific notation to standard
    notation. The exponent must be represented in base-10 and delineated by an 'e'. Leading and
    trailing zeros will be stripped from the output.

    e.g.
    standard_notation('000452AEF.FB3162600000e-7') -> '0.0452AEFFB31626'
    standard_notation('000452AEF.FB3162600000e9') -> '452AEFFB3162600'

    number_str  - (string) A string representing a number in scientific notation.

    output      - (string) A string representing the same number in standard notation.
    """
    assert type(number_str) is str
    if "e" in number_str:
        split_str = number_str.split("e")
        value_str = split_str[0] #.replace(".", "")
        exponent = int(split_str[1])

        if "." in value_str:
            integer_part, fractional_part = value_str.split(".")
        else:
            integer_part = value_str
            fractional_part = ""

        number_str = integer_part + fractional_part

        leading_zeros = 0
        for digit in number_str:
            if digit == "0":
                leading_zeros += 1
            else:
                break

        trailing_zeros = 0
        for digit in number_str[::-1]:
            if digit == "0":
                trailing_zeros += 1
            else:
                break

        number_str = number_str[leading_zeros:len(number_str)-trailing_zeros]

        # Index of the digit to insert the decimal point after
        decimal_point_index = len(integer_part) - leading_zeros + exponent - 1

        if decimal_point_index <= -1:
            number_str = "0." + "0"*(abs(decimal_point_index)-1) + number_str
        elif (decimal_point_index >= 0) and (decimal_point_index < len(number_str)-1):
            number_str = number_str[:decimal_point_index+1] + "." + number_str[decimal_point_index+1:]
        elif decimal_point_index >= len(number_str)-1:
            number_str += "0"*(decimal_point_index-len(number_str)+1)

    return number_str

def cartprod(*arrays):
    """
    Generates the Cartesian product of any number of input lists.
    The input lists can contain elements of any data type.

    arrays[n]  - (list) List of any length.

    output     - (list of lists) The Cartesian product of the lists in 'arrays'.
    """
    for element in arrays:
        try:
            element[len(element)-1] # check that the elements are indexable
        except:
            raise ValueError

    arrays = arrays[::-1]
    output = [ [element] for element in arrays[0] ]
    for array in arrays[1:]:
        new_output = []
        for array_element in array:
            for output_array in output:
                new_output.append( [array_element] + output_array )
        del output
        output = new_output
    return output

def feynman_point_probability( n, r, b, prob_library={}, max_recursion_depth=20 ):
    """
    Given a randomly sampled base-b number that is n digits long, calculates the probability of a digit occuring r times in succession, using a recursion relation.

    e.g. The probability of '0000000' occuring within the first 16 digits of pi (assuming the digits of pi are 'normal'/randomly sampled).
    3.1415928500000000923429358...

    n                       - (int) The number of digits comprising the number.
    r                       - (int) The number of times a digit occurs successively.
    b                       - (int) The base of the number.
    prob_library            - (dict) A dictionary where the output of p(n,r,b) is saved so that it
                              doesn't need to be recalculated.
    max_recursion_depth     - (int) This prevents recursion problems: Calculation of p(n,r,b)
                              requires p(n-1,r,b) to be calculated, so if n is larger than the
                              maximum recursion depth N in Python, then this function is guaranteed
                              to fail. By saving the output of p(n,r,b) for n < N to prob_library,
                              we can safely calculate p(n,r,b) for arbitrarily large n.

    output                  - (float) The probability from 0-1 of the event occurring.
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

def generate_primes():
    """
    Sieve of Eratosthenes
    Code by David Eppstein, UC Irvine, 28 Feb 2002
    http://code.activestate.com/recipes/117119/

    Generate an infinite sequence of prime numbers.

    output  - (int) Yields a single prime number per iteration.
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

def generate_rationals(n=0):
    """
    Generates every positively valued rational number.
    This generator uses the recursive method described in Calkin & Wilf 1999.

    n       - (int) The number of rational numbers to generate.

    output  - (2-tuple of integers) Yields a single 2-tuple once per iteration, where the first
              element is the numerator, and the second element is the denominator.
    """
    vals = [1,1]
    i = 1
    while i <= n:
        yield (vals[i-1], vals[i]) # (numerator, denominator)
        vals.extend([ vals[i-1]+vals[i], vals[i] ])
        i += 1

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
