import gmpy2, math, re, sys, random, time
import numpy as np

from user_tools import Console, task, subset_colormap, cartprod, save_table_to_txt, read_table_from_txt, concatenate_tables
from tools.math_tools import standard_notation, feynman_point_probability, generate_rationals
from baseconvert import base
from ANSI_tools import color_text, ANSI_RESET, rgb_to_ANSICtrl

import matplotlib as mpl
import matplotlib.pyplot as plt

num_len_base10 = 2500 # maximum number of base-10 digits to consider (this will result in more digits in bases <10, and less digits in bases >10)

gmpy2.get_context().allow_complex = True
ln10ln2 = 3.3219280948873626 # math.log(10)/math.log(2)
num_len_base2 = int( math.log(10)/math.log(2) * ( num_len_base10 + 1 ) )
gmpy2.get_context().precision = num_len_base2 # precision is specfied in base-2

prob_library = {}

log10prob_limit = -4 # minimum probability necessary for the entire number to be flagged as interesting and printed to the terminal.
# str_length_limit = 6
line_length = 128 # maximum number of digits to print in a row to the terminal before a new line.
# order_save_freq = 200

def make_frac_label(Q, as_leading_coeff=True):
    """
    Converts a rational number represented as a 2-tuple of integers to a string.
    e.g.
    (2,3) -> "2/3"
    (3,1) -> "3"
    (1,1) -> "" if as_leading_coeff=True else "1"
    """
    a,b = Q
    if b == 1 and a == 1:
        label = "" if as_leading_coeff else "1"
    elif b == 1:
        label = str(a)
    else:
        label = str(a) + "/" + str(b)
    return label

def make_exp_label(Q):
    label = make_frac_label(Q)
    label = f"^{label}" if label != "" else label
    return label

def make_label(base,Q,const,q):
    qlabel = make_frac_label(Q)
    nlabel = make_exp_label(q)
    label = f"{qlabel} {const}{nlabel}, base {base}"
    return label

# Let's define a set of irrational constants to examine.
constant_paths = {
    "pi": "constants/pi.txt"
    ,"e": "constants/e.txt"
    ,"phi": "constants/phi.txt"
}

constants = {}
for constant in ("pi","e","phi"):
    path = constant_paths[constant]
    raw_data = open(path,"rt").read()
    constants[constant] = gmpy2.mpfr(raw_data[:num_len_base10])

constants["e^pi"] = gmpy2.exp( constants["pi"] ) # Gelfond's constant
constants["e^pi-pi"] = gmpy2.exp(pi)-pi # Extremely close to 20, for an unknown reason
constants["ln(pi)"] = gmpy2.log(pi)
constants["pi-e"] = constants["pi"] - constants["e"]
constants["ln(2)"] = gmpy2.log(2)
constants["ln(3)"] = gmpy2.log(3)

constant_names = list(constants.keys())
exponents = []
for a,b in generate_rationals(3):
    exponents.extend([ (a,b), (-a,b) ])
coefficients = [ (a,b) for a,b in generate_rationals(4) ]
bases = [10,11,5]

constants_to_check = cartprod( constant_names, exponents, coefficients, bases )


constants_data = {} # Save the previous computed constants to a dictionary, so that they can be referenced for conversion into other bases.
def get_constant_value( const_key, exp, coeff ):
    """
    Computes the value of the constant.

    const_key   - (string) The key associated with the constant in the dictionary 'constants'.
    exp         - (2-tuple of integers) A 2-tuple representing a rational number. The first element
                  is the numerator and the second element is the denominator.
    coeff       - (2-tuple of integers) A 2-tuple representing a rational number. The first element
                  is the numerator and the second element is the denominator.

    output      - (gmpy2.mpfr) The numerical value of the constant.
    """
    if const not in constants_data:
        constants_data[const] = {}
    if exp not in constants_data[const]:
        constants_data[const][exp] = {}
    if coeff not in constants_data[const][exp]:
        constants_data[const][exp][coeff] = gmpy2.mpq(*coeff)*(constants[const]**gmpy2.mpq(*exp))
    return constants_data[const][exp][coeff]

# Generate constants for the product pi^n * e^m
exponent_pairs = cartprod( exponents, exponents )
for q1, q2 in exponent_pairs:
    key = f"pi{make_exp_label(q1)} * e{make_exp_label(q2)}"
    constants[key] = constants["pi"]**gmpy2.mpq(*q1) * constants["e"]**gmpy2.mpq(*q2)

def colorize_digit_sequences(number_string, b, log10prob_limit):
    """
    Given a string of randomly sampled digits, colorizes recurring sequences of digits according to their probability of occurring within the first N digits.

    number_string   - (string) The number's digit expansion in any base. There is no limit to the
                      length of this string.
    b               - (int) The base of the number to be colorized.
    log10prob_limit - (int) A negative integer representing the base-10 logarithm of the minimum
                      probability to coloring. More negative integers will result in only extremely
                      improbable digit sequences being colored.
    """

    norm = mpl.colors.Normalize(vmin=log10prob_limit, vmax=0)
    cmap = subset_colormap("YlOrRd_r", new_min=0, new_max=0.7)

    # Match all occurrences of digits repeating themselves twice or more successively.
    integer_part, fractional_part = number_string.split(".")
    matcher = re.compile(r'(.)\1*')
    consec_num_groups = [ match.group() for match in matcher.finditer(number_string) ]

    # Rebuild the number string, colorizing groups of repeated digits according to their probability of occurring.
    str_array = [integer_part,"."]

    index = 0 # Increments once per digit.
    log10min_prob = 0 # Records the minimum probability identified.
    min_prob_index = 0 # Records the index of the most imporbable sequence of repeated digits.
    for i in range(len(consec_num_groups)):
        group = consec_num_groups[i]
        len_group = len(group)
        digit = group[0] # All digits in the group are identical.

        if len_group == 1:
            # Single digits need not be colorized
            str_array.append(group)
            index += 1
            continue
        elif (index == 0) and (integer_part in ("0","")) and (digit == "0"):
            # It's just a small number with some leading zeros; not worth flagging as improbable.
            for e in group:
                str_array.append(e)
            index += len_group
            continue

        # Calculate the probability of this sequence of digits occurring.
        n = index + len_group # number of digits up to the last digit in the sequence.
        r = len_group
        prob = feynman_point_probability(n,r,b,prob_library=prob_library)
        log10prob = math.log10( prob )

        if log10prob < log10min_prob:
            # Update index and probability of least probable digit sequence.
            log10min_prob = log10prob
            min_prob_index = index

        # Append colorized digit sequence to str_array
        rgb_color = cmap(norm( log10prob ))[:3]
        ANSI_COLOR = rgb_to_ANSICtrl( rgb_color )
        str_array.extend( [ANSI_COLOR + digit] + [digit]*(len_group-2) + [digit + ANSI_RESET] )

        # str_array.append( ANSI_COLOR + digit )
        # for _ in range(len_group-2):
        #     str_array.append(digit)
        # str_array.append( digit + ANSI_RESET )

        index += len_group

    return str_array, log10min_prob, min_prob_index

def print_str_array(str_array, line_length, indent=6):
    number_of_lines = (len(str_array)-1)//line_length + 1
    for i in range(0, number_of_lines):
        index_i = i*line_length
        index_f = (i+1)*line_length
        line = ''.join(str_array[index_i:index_f])
        Console.out(" "*indent + f"{line}")

# Following code gets run for each constant to analyze.

ln10lnb = math.log(10)/math.log(b)
num_len_baseb = int( ln10lnb * ( num_len_base10 + 10 ) )

val_str_base_10 = standard_notation( str( constant_value ) )
val_str_base_b = base(val_str, input_base=10, output_base=b, max_depth=num_len_baseb, string=True, recurring=False) # convert number string to base b

str_array, log10min_prob, min_prob_index = colorize_digit_sequences(val_str_base_b, b, log10prob_limit)


print_str_array(str_array, line_length)

task(
    run,
    start_text="Computing digits",
    end_text="Computed digits",
    fail_text="Failed to compute digits",
    pass_update_text=True,
    exit_on_fail=True
)()


    label = make_label(b,Q,const,n)

    # compute probability of digits repeats, skip if above threshold
    if (log10min_prob > log10prob_limit): # and (max_len_group < str_length_limit):
        uninteresting_orders_table["const"].append(const)
        uninteresting_orders_table["base"].append(b)
        uninteresting_orders_table["exp"].append(n_qstr)
        uninteresting_orders_table["coeff"].append(Q_qstr)
        uninteresting_orders_table["interesting?"].append("No")
        Console.out(f"{label} (log10(min prob): {str(log10min_prob)}):")
        continue

    Console.out("")
    Console.out(f"{label} (log10(min prob): {str(log10min_prob)}):")



# def run(**kwargs):
#
#     update_text = kwargs["update_text"]
#
#     constants = {}
#     # Search for sequences in pi in any base
#
#     digit_data_column_titles = ["const", "base", "exp", "coeff", "index", "log10/logb * i/n", "# of digits", "log10(prob)", "string"]
#     try:
#         update_text("Loading digit data")
#         messages_table = read_table_from_txt("digit_data.txt")
#     except:
#         messages_table = { column_title: [] for column_title in digit_data_column_titles }
#
#     len_orders = len(newly_generated_orders)
#     # random.shuffle(newly_generated_orders)
#     m=1
#     for order in newly_generated_orders:
#         const, b, n_qstr, Q_qstr = order
#         b = int(b)
#         n = qstr_to_tuple(n_qstr)
#         Q = qstr_to_tuple(Q_qstr)
#
#         if m%order_save_freq == 0:
#             update_text("Saving orders table")
#
#             # combine interesting and uninteresting orders into one table
#
#             orders_table = concatenate_tables(interesting_orders_table, uninteresting_orders_table)
#
#             save_table_to_txt( "computed_orders.txt", orders_table, column_titles=orders_column_titles, column_padding=3 )
#
#         m+=1
#
#         update_text(f"Processing order #{m}/{len_orders}: {make_frac_label(*Q)} {const}{make_exp_label(*n)} in base {b}")
#
#         if const not in constants_data.keys():
#             constants_data[const] = {}
#         if "value" not in constants_data[const].keys():
#
#             const_value = generate_constant_value(const)
#
#             constants_data[const]["value"] = const_value
#         if "exponents" not in constants_data[const].keys():
#             constants_data[const]["exponents"] = {}
#         if n not in constants_data[const]["exponents"].keys():
#             exponent_value = gmpy2.mpq(*n)
#             const_value = constants_data[const]["value"]**exponent_value
#             constants_data[const]["exponents"][n] = const_value
#         else:
#             const_value = constants_data[const]["exponents"][n]
#
#         norm = mpl.colors.Normalize(vmin=-3, vmax=0) #log10prob_limit, vmax=0)
#         cmap = subset_colormap("YlOrRd_r", new_min=0, new_max=0.7)
#
#         ln10lnb = math.log(10)/math.log(b)
#         num_len_baseb = int( ln10lnb * ( num_len_base10 + 10 ) )
#
#         # for Q in QRs:
#         Q_value = gmpy2.mpq(*Q)
#         p2 = standard_notation( str( Q_value * const_value ) )
#
#         # Convert string to standard notation if it's formatted in scientific notation
#
#         val = base(p2, input_base=10, output_base=b, max_depth=num_len_baseb, string=True, recurring=False)
#         # colorize repeats
#
#         lead, tail = val.split(".")
#         matcher = re.compile(r'(.)\1*')
#         consec_num_groups = [ match.group() for match in matcher.finditer(tail) ]
#
#         str_array = [lead,"."]
#
#         index = 0
#         min_prob_index = 0
#         log10min_prob = 0
#         max_len_group = 1
#         for i in range(len(consec_num_groups)):
#             group = consec_num_groups[i]
#             len_group = len(group)
#             digit = group[0]
#
#             if len_group == 1:
#                 str_array.append(group)
#                 index += 1
#                 continue
#             elif (index == 0) and (val[0] == "0") and (digit == "0"):
#                 # it's just a small number with leading zeros
#                 for e in group:
#                     str_array.append(e)
#                 index += len_group
#                 continue
#             if len_group > max_len_group:
#                 max_len_group = len_group
#
#             # rgb_color = hexcolor_to_rgb(color_conv[ len_repeat if len_repeat <= max(list(color_conv.keys())) else max(list(color_conv.keys())) ])
#
#
#             # Calculate the probability of this number sequence occuring
#             N = index + len_group # number of digits preceding and including the repeating string
#             k = len_group
#             prob = feynman_point_probability(N,k,b,prob_library=prob_library)
#             log10prob = math.log10( prob )
#             #log10prob = math.log10(b**(-len_group))
#
#             # plus 1 if the digit is not 0 or 9 (bc close to zero is special lol)
#             # if digit not in ("0","9"):
#             #     log10prob += 1
#
#             if log10prob < log10min_prob:
#                 log10min_prob = log10prob
#                 min_prob_index = index
#
#             # build string to print to console
#             rgb_color = cmap(norm( log10prob ))[:3]
#             ANSI_COLOR = rgb_to_ANSICtrl( rgb_color )
#             str_array.append( ANSI_COLOR + digit )
#             for _ in range(len_group-2):
#                 str_array.append(digit)
#             str_array.append( digit + ANSI_RESET )
#
#             index += len_group
#
#         label = make_label(b,Q,const,n)
#






print("Done! :)")
