import gmpy2, math
from tools.math_tools import generate_rationals, cartprod
from tools.label_tools import make_exp_label
from settings import num_len_base10

num_len_base2 = int( math.log(10)/math.log(2) * ( num_len_base10 + 1 ) )
gmpy2.get_context().precision = num_len_base2 # precision is specfied in base-2

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

# constant_values["e^pi"] = gmpy2.exp(constant_values["pi"]) # Gelfond's constant
# constant_values["e^pi-pi"] = gmpy2.exp(constant_values["pi"])-constant_values["pi"] # Extremely close to 20, for an unknown reason
# constant_values["ln(pi)"] = gmpy2.log(constant_values["pi"])
# constant_values["ln(ln(pi))"] = gmpy2.log(gmpy2.log(constant_values["pi"]))
# constant_values["pi-e"] = constant_values["pi"] - constant_values["e"]
# constant_values["ln(2)"] = gmpy2.log(2)
# constant_values["ln(3)"] = gmpy2.log(3)
#
# constant_keys = list(constant_values.keys())
# exponents = []
# for a,b in generate_rationals(3):
#     exponents.extend([ (a,b), (-a,b) ])
# coefficients = [ (a,b) for a,b in generate_rationals(3) ]
# bases = [10,11,5]
#
# constants_to_check = cartprod( bases, constant_keys, exponents, coefficients )
#
# output_file = "output/output1.txt"



###########################################
#### Uncomment to generate output2.txt ####
###########################################

# constant_values["2"] = gmpy2.mpz(2)
# constant_values["3"] = gmpy2.mpz(3)
# constant_values["5"] = gmpy2.mpz(5)
# constant_values["7"] = gmpy2.mpz(7)
#
# constant_keys = ["2","3","5","7"]
# exponents = [ (-1,n) for n in (2,3,4,5) ]
# coefficients = [ (a,b) for a,b in generate_rationals(7) ]
# bases = [10,11,5]
#
# output_file = "output/output2.txt"
#
# constants_to_check = cartprod( bases, constant_keys, exponents, coefficients )



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

constant_keys = ["pi"]
exponents = [(-1,2)]
bases = [11]
coefficients = []
for a,b in generate_rationals(100):
    if a%11!=0 and b%11!=0:
        coefficients.append( (a,b) ) #coefficients.append( (10*a,49*b) )
constants_to_check = cartprod( bases, constant_keys, exponents, coefficients )

output_file = "output/output4.txt"
