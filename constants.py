import gmpy2, math
from tools.math_tools import generate_rationals, cartprod
from tools.label_tools import make_exp_label


log10prob_limit = -3 # minimum probability necessary for the entire number to be flagged as interesting and printed to the terminal.
line_length = 128 # maximum number of digits to print to the terminal before a new line.
num_len_base10 = 2400 # maximum number of base-10 digits to consider (this will result in more digits in bases <10, and less digits in bases >10)


# gmpy2 precision is specfied in base-2
num_len_base2 = int( math.log(10)/math.log(2) * ( num_len_base10 + 1 ) )
gmpy2.get_context().precision = num_len_base2


constant_paths = {
    "pi": "constants/pi.txt"
    ,"e": "constants/e.txt"
    ,"phi": "constants/phi.txt"
}

constant_values = {}
for constant in ("pi","e","phi"):
    path = constant_paths[constant]
    raw_data = open(path,"rt").read()
    constant_values[constant] = gmpy2.mpfr(raw_data[:num_len_base10])



###########################################
#### Uncomment to generate output1.txt ####
###########################################

constant_values["e^pi"] = gmpy2.exp(constant_values["pi"]) # Gelfond's constant
constant_values["e^pi-pi"] = gmpy2.exp(constant_values["pi"])-constant_values["pi"] # Extremely close to 20, for an unknown reason
constant_values["ln(pi)"] = gmpy2.log(constant_values["pi"])
constant_values["ln(ln(pi))"] = gmpy2.log(gmpy2.log(constant_values["pi"]))
constant_values["pi-e"] = constant_values["pi"] - constant_values["e"]
constant_values["ln(2)"] = gmpy2.log(2)
constant_values["ln(3)"] = gmpy2.log(3)

constant_keys = list(constant_values.keys())
exponents = []
for a,b in generate_rationals(3):
    exponents.extend([ (a,b), (-a,b) ])
coefficients = [ (a,b) for a,b in generate_rationals(3) ]
bases = [10,11,5]

constants_to_check = cartprod( bases, constant_keys, exponents, coefficients )

output_file = "output/output1.txt"



###########################################
#### Uncomment to generate output2.txt ####
###########################################

# from tools.math_tools import generate_primes
#
# constant_keys = []
# for i, p in zip(range(100),generate_primes()):
#     constant_key = str(p)
#     constant_keys.append( constant_key )
#     constant_values[constant_key] = gmpy2.mpz(p)
# exponents = []
# for a,b in generate_rationals(14):
#     if b != 1:
#         exponents.append( (a,b) )
# coefficients = [ (1,1) ]
# bases = [10,11,16]
#
# constants_to_check = cartprod( bases, constant_keys, exponents, coefficients )
#
# output_file = "output/output2.txt"



###########################################
#### Uncomment to generate output3.txt ####
###########################################

# exponents = []
# for a,b in generate_rationals(3):
#     exponents.extend([ (a,b), (-a,b) ])
# coefficients = [ (a,b) for a,b in generate_rationals(3) ]
#
# # Generate constants for the product e^n * pi^m
# exponent_pairs = cartprod( exponents, exponents )
# for n, m in exponent_pairs:
#     key = f"e{make_exp_label(n)} * pi{make_exp_label(m)}"
#     constant_values[key] = constant_values["e"]**gmpy2.mpq(*n) * constant_values["pi"]**gmpy2.mpq(*m)
#
# constant_keys = list(constant_values.keys())[3:]
# exponents = [(1,1)]
# bases = [10,11,5]
#
# constants_to_check = cartprod( bases, constant_keys, exponents, coefficients )
#
# output_file = "output/output3.txt"



###########################################
#### Uncomment to generate output4.txt ####
###########################################

# constant_keys = ["pi"]
# exponents = [(-1,2)]
# bases = [11]
# coefficients = [(1,84**2)]
# for a,b in generate_rationals(100):
#     if a%11!=0 and b%11!=0:
#         # We can skip rational numbers with factors of 11 in the numerator/denominator, since they just result in a left/right translation of the digits.
#         coefficients.append( (a,b) )
#
# constants_to_check = cartprod( bases, constant_keys, exponents, coefficients )
#
# output_file = "output/output4.txt"
