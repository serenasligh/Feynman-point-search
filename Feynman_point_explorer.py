import gmpy2, math, re, sys, random, time
import numpy as np

from user_tools import Console, task, subset_colormap, cartprod, save_table_to_txt, read_table_from_txt, concatenate_tables, standard_notation, gen_primes
from baseconvert import base
from ANSI_tools import color_text, ANSI_RESET, rgb_to_ANSICtrl

import matplotlib as mpl
import matplotlib.pyplot as plt

constant_paths = {
    "pi": "library/constants/pi.txt"
    ,"e": "library/constants/e.txt"
    ,"phi": "library/constants/phi.txt"
}

constants_data = {}
prob_library = {}

ln10ln2 = 3.3219280948873626 # math.log(10)/math.log(2)

num_len_base10 = 2000
num_len_base2 = int( ln10ln2 * ( num_len_base10 + 1 ) )

gmpy2.get_context().allow_complex = True
gmpy2.get_context().precision = num_len_base2

log10prob_limit = -3
# str_length_limit = 6
line_length = 128
order_save_freq = 200

def feynman_point_probability( n, r, b, prob_library={}, max_recursion_depth=20 ):
    """
    Given a base-b number that is n digits long, calculates the probability of a digit occuring r times in succession, using a recursion relation.

    e.g. The probability of '0000000' occuring within the first 16 digits of pi (assuming the digits of pi are 'normal'/randomly sampled).
    3.1415928500000000923429358...

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
                prob_library[(n,r,b)] = p_n # Save the output of the calculation to prob_library
                return p_n

    # We can avoid recursion problems by filling out prob_library
    i_max = n//max_recursion_depth + 1
    i=1
    while i < i_max:
        p(max_recursion_depth*i,r,b)
        i += 1
    return p(n,r,b)

def make_frac_label(a,b,for_db=False):
    if b == 1 and a == 1:
        label = "" if not for_db else "1"
    elif b == 1:
        label = str(a)
    else:
        label = str(a) + "/" + str(b)
    return label

def make_exp_label(a,b):
    label = make_frac_label(a,b)
    label = f"^({label})" if label != "" else label
    return label

def make_label(base,Q,const,n):
    qlabel = make_frac_label(*Q)
    nlabel = make_exp_label(*n)
    label = f"{qlabel} {const}{nlabel}, base {base}"
    return label

def generate_rationals(Q_limit=None, Q_denoms=None, include_negative=False, include_integers=True):
    # For each denominator denom in Q_denoms, returns a list of all reduced rational numbers with denom in the denominator that are smaller in magnitude than Q_limit
    Qs = []
    non_factors = {1:[1]}
    for denom in Q_denoms:
        if denom==1:
            continue
        nf = []
        for N in list(range(1,denom)):
            if math.gcd(denom,N) == 1:
                nf.append(N)
        non_factors[denom] = nf

    for denom in Q_denoms:
        i=0
        while i < Q_limit:
            Qs += [ (N + i*denom,denom) for N in non_factors[denom] ]
            i += 1

    if not include_integers:
        new_Qs = []
        for Q in Qs:
            if Q[1] != 1:
                new_Qs.append(Q)
        Qs = new_Qs

    if include_negative:
        neg_Qs = [ (-n[0],n[1]) for n in Qs ]
        new_Qs = []
        for i in range(len(Qs)):
            new_Qs += [ Qs[i], neg_Qs[i] ]
        Qs = new_Qs

    return Qs

def qstr_to_tuple(string):
    num_and_denom = string.split("/")
    if len(num_and_denom) == 1:
        a = num_and_denom[0]
        b = "1"
    else:
        a,b = num_and_denom
    a,b = tuple(map(int,(a,b)))
    return (a,b)

# First, let's generate a list of "orders". Each order generates an irrational number which we want to investigate. Orders can be saved in a list (orders.txt) so that we don't have to spend time recomputing any of the many many numbers that we've already investigated.
primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997]
#
# def make_primes(max_i=1, **kwargs):
#     update_text = kwargs["update_text"] if "update_text" in kwargs else lambda x: None
#     primes = []
#     i=0
#     prime_gen = gen_primes()
#     while i < max_i:
#         time.sleep(0.0005)
#         if i%int(max_i/100.)==0:
#             update_text(f"Generating primes [{int((100.*i)/max_i)}%]")
#         p = next(prime_gen)
#         primes.append(p)
#         i += 1
#     return primes
#
#
# primes = task(
#     make_primes,
#     start_text="Generating primes",
#     end_text="Generated primes",
#     fail_text="Failed to generate primes",
#     exit_on_fail=True,
#     pass_update_text=True,
# )(max_i=1000)

primes = [ p for i,p in zip(range(1,100000),gen_primes())]
# print(primes)
# sys.exit()

def generate_orders():
    orders = []

    # Adjust the parameters in this function to generate new numbers to investigate

    const = "pi"
    Qs = [ (10, ( 7056 * p ) ) for p in primes ]
    random.shuffle(Qs)
    # Qs = [ (100,p) for p in primes ]


    # Q_denoms = []
    # for i,j,k in cartprod(range(0,5),range(0,3),range(0,3)):
    #     v = 2**i * 3**j * 7**k
    #     # if v <= 500:
    #     Q_denoms.append(v)
    #
    # Qs = generate_rationals(Q_limit=1, Q_denoms=Q_denoms)
    ns = [(-1,2)] #generate_rationals(Q_limit=3, Q_denoms=[1,2], include_negative=True)
    # ns += generate_rationals(Q_limit=1, Q_denoms=[3,4], include_negative=True)
    bases = [11] #,7,17,19,13,21,6,8,14,15,18,20,4]
    orders += cartprod( [const], bases, list(map(lambda x: make_frac_label(*x, for_db=True),ns)), list(map(lambda x: make_frac_label(*x, for_db=True),Qs)) )

    # const = "e"
    # Qs = generate_rationals(Q_limit=2, Q_denoms=[1,2,3,4,5,6,7,8,9,10,11,12])
    # ns = generate_rationals(Q_limit=3, Q_denoms=[1,2], include_negative=True)
    # ns += generate_rationals(Q_limit=1, Q_denoms=[3,4], include_negative=True)
    # bases = [12] #[10,11,16,9,5,12] #,7,17,19,13,21,6,8,14,15,18,20,4]
    # orders += cartprod( [const], bases, list(map(lambda x: make_frac_label(*x, for_db=True),ns)), list(map(lambda x: make_frac_label(*x, for_db=True),Qs)) )

    # default investigation palette
    # consts = ["(2pi-1)*e^pi", "pi*(pi-1)", "pi*e^pi", "e^pi-pi", "ln(2pi-1)", "ln(pi-1)", "ln((2pi-1)*e^pi)", "ln(pi*(pi-1))", "ln(pi*e^pi)", "ln(e^pi-pi)"] # "ln(pi)", "(e^pi)",#, "ln(2)", "ln(3)", "phi", "2", "3" ]
    # for const in consts:
    #     Qs = generate_rationals(Q_limit=2, Q_denoms=[1,2,3])
    #     include_integer_exponents = const not in ("2","3")
    #     ns = generate_rationals(Q_limit=2, Q_denoms=[1,2], include_negative=True, include_integers=include_integer_exponents)
    #     bases = [10,11,16,9,5] #,7,17,19,13,21,6,8,14,15,18,20,4]
    #     orders += cartprod( [const], bases, list(map(lambda x: make_frac_label(*x, for_db=True),ns)), list(map(lambda x: make_frac_label(*x, for_db=True),Qs)) )

    return orders

generated_orders = task(
    generate_orders,
    start_text="Generating orders",
    end_text="Generated orders",
    fail_text="Failed to generate orders",
    exit_on_fail=True
)()

# if new orders are generated by generate_orders (that aren't in orders.txt) then they'll be appended to orders.txt

orders_table = task(
    read_table_from_txt,
    start_text="Loading orders",
    end_text="Loading orders",
    fail_text="Failed to load orders",
    exit_on_fail=False
)("computed_orders.txt")

orders_column_titles = ["const", "base", "exp", "coeff", "interesting?"]
if orders_table is None:
    orders_table = { column_title: [] for column_title in orders_column_titles }

def filter_orders(generated_orders, orders_table):
    # Now compare the loaded orders to the generated orders
    loaded_orders = []
    for i in range(len(orders_table["interesting?"])):
        order = [
            orders_table["const"][i]
            ,orders_table["base"][i]
            ,orders_table["exp"][i]
            ,orders_table["coeff"][i]
        ]
        loaded_orders.append(order)

    newly_generated_orders = []
    for generated_order in generated_orders:
        if generated_order in loaded_orders:
            # We've already computed this order
            continue
        else:
            newly_generated_orders.append( generated_order )
    return newly_generated_orders

# newly_generated_orders = task(
#     filter_orders,
#     start_text="Filtering orders",
#     end_text="Filtered orders",
#     fail_text="Failed to filter orders",
#     exit_on_fail=True
# )(generated_orders, orders_table)
newly_generated_orders = generated_orders


def sort_orders(orders_table):
    interesting_orders_table = { column_title: [] for column_title in orders_column_titles }
    uninteresting_orders_table = { column_title: [] for column_title in orders_column_titles }

    for i in range(len(orders_table["interesting?"])):
        order = [
            orders_table["const"][i]
            ,orders_table["base"][i]
            ,orders_table["exp"][i]
            ,orders_table["coeff"][i]
        ]
        if orders_table["interesting?"][i] == "Yes":
            for j in range(len(orders_column_titles)-1):
                interesting_orders_table[orders_column_titles[j]].append( order[j] )
                interesting_orders_table["interesting?"].append("Yes")
        elif orders_table["interesting?"][i] == "No":
            for j in range(len(orders_column_titles)-1):
                uninteresting_orders_table[orders_column_titles[j]].append( order[j] )
                uninteresting_orders_table["interesting?"].append("No")

    return interesting_orders_table, uninteresting_orders_table

interesting_orders_table, uninteresting_orders_table = task(
    sort_orders,
    start_text="Sorting orders",
    end_text="Sorted orders",
    fail_text="Failed to sort orders",
    exit_on_fail=True
)(orders_table)

del orders_table

# def save_orders(orders):
#
#     for order in orders:
#         for i in range(len(orders_column_titles)-1):
#             orders_table[orders_column_titles[i]].append( str(order[i]) )
#             orders_table["interesting?"].append("Unknown")
#
#     save_table_to_txt( "orders.txt", orders_table, column_titles=orders_column_titles, column_padding=3 )
#
# task(
#     save_orders,
#     start_text="Saving orders",
#     end_text="Saved orders",
#     fail_text="Failed to save orders",
#     exit_on_fail=True
# )()

def generate_constant_value(const):
    if const in ("pi","e","phi"):
        path = constant_paths[const]
        raw_data = open(path,"rt").read()
        return gmpy2.mpfr(raw_data[:num_len_base10])

    pi_path = constant_paths["pi"]
    pi_raw_data = open(pi_path,"rt").read()
    pi = gmpy2.mpfr(pi_raw_data[:num_len_base10])
    if const == "ln(pi)":
        return gmpy2.log(pi)
    elif const == "(e^pi)":
        return gmpy2.exp(pi)
    elif const == "(2pi-1)":
        return (2*pi-1)
    elif const == "(pi-1)":
        return (pi-1)
    elif const == "(2pi-1)*e^pi":
        return (2*pi-1)*gmpy2.exp(pi)
    elif const == "pi*(pi-1)":
        return pi*(pi-1)
    elif const == "pi*e^pi":
        return pi*gmpy2.exp(pi)
    elif const == "e^pi-pi":
        return gmpy2.exp(pi)-pi
    elif const == "ln(2pi-1)":
        return gmpy2.log(2*pi-1)
    elif const == "ln(pi-1)":
        return gmpy2.log(pi-1)
    elif const == "ln((2pi-1)*e^pi)":
        return gmpy2.log( (2*pi-1)*gmpy2.exp(pi) )
    elif const == "ln(pi*(pi-1))":
        return gmpy2.log( pi*(pi-1) )
    elif const == "ln(pi*e^pi)":
        return gmpy2.log( pi*gmpy2.exp(pi) )
    elif const == "ln(e^pi-pi)":
        return gmpy2.log( gmpy2.exp(pi)-pi )
    elif const == "ln(2)":
        return gmpy2.log(2)
    elif const == "ln(3)":
        return gmpy2.log(3)
    elif const == "2":
        return gmpy2.mpz(2)
    elif const == "3":
        return gmpy2.mpz(3)


def run(**kwargs):

    update_text = kwargs["update_text"]

    constants = {}
    # Search for sequences in pi in any base

    messages_column_titles = ["const", "base", "exp", "coeff", "index", "log10/logb * i/n", "# of digits", "log10(prob)", "string"]
    try:
        update_text("Loading messages")
        messages_table = read_table_from_txt("digit_messages.txt")
    except:
        messages_table = { column_title: [] for column_title in messages_column_titles }

    len_orders = len(newly_generated_orders)
    # random.shuffle(newly_generated_orders)
    m=1
    for order in newly_generated_orders:
        const, b, n_qstr, Q_qstr = order
        b = int(b)
        n = qstr_to_tuple(n_qstr)
        Q = qstr_to_tuple(Q_qstr)

        if m%order_save_freq == 0:
            update_text("Saving orders table")

            # combine interesting and uninteresting orders into one table

            orders_table = concatenate_tables(interesting_orders_table, uninteresting_orders_table)

            save_table_to_txt( "computed_orders.txt", orders_table, column_titles=orders_column_titles, column_padding=3 )

        m+=1

        update_text(f"Processing order #{m}/{len_orders}: {make_frac_label(*Q)} {const}{make_exp_label(*n)} in base {b}")

        if const not in constants_data.keys():
            constants_data[const] = {}
        if "value" not in constants_data[const].keys():

            const_value = generate_constant_value(const)

            constants_data[const]["value"] = const_value
        if "exponents" not in constants_data[const].keys():
            constants_data[const]["exponents"] = {}
        if n not in constants_data[const]["exponents"].keys():
            exponent_value = gmpy2.mpq(*n)
            const_value = constants_data[const]["value"]**exponent_value
            constants_data[const]["exponents"][n] = const_value
        else:
            const_value = constants_data[const]["exponents"][n]

        norm = mpl.colors.Normalize(vmin=-3, vmax=0) #log10prob_limit, vmax=0)
        cmap = subset_colormap("YlOrRd_r", new_min=0, new_max=0.7)

        ln10lnb = math.log(10)/math.log(b)
        num_len_baseb = int( ln10lnb * ( num_len_base10 + 10 ) )

        # for Q in QRs:
        Q_value = gmpy2.mpq(*Q)
        p2 = standard_notation( str( Q_value * const_value ) )

        # Convert string to standard notation if it's formatted in scientific notation

        val = base(p2, input_base=10, output_base=b, max_depth=num_len_baseb, string=True, recurring=False)
        # colorize repeats

        lead, tail = val.split(".")
        matcher = re.compile(r'(.)\1*')
        consec_num_groups = [ match.group() for match in matcher.finditer(tail) ]

        str_array = [lead,"."]

        index = 0
        min_prob_index = 0
        log10min_prob = 0
        max_len_group = 1
        for i in range(len(consec_num_groups)):
            group = consec_num_groups[i]
            len_group = len(group)
            digit = group[0]

            if len_group == 1:
                str_array.append(group)
                index += 1
                continue
            elif (index == 0) and (val[0] == "0") and (digit == "0"):
                # it's just a small number with leading zeros
                for e in group:
                    str_array.append(e)
                index += len_group
                continue
            if len_group > max_len_group:
                max_len_group = len_group

            # rgb_color = hexcolor_to_rgb(color_conv[ len_repeat if len_repeat <= max(list(color_conv.keys())) else max(list(color_conv.keys())) ])


            # Calculate the probability of this number sequence occuring
            N = index + len_group # number of digits preceding and including the repeating string
            k = len_group
            prob = feynman_point_probability(N,k,b,prob_library=prob_library)
            log10prob = math.log10( prob )
            #log10prob = math.log10(b**(-len_group))

            # plus 1 if the digit is not 0 or 9 (bc close to zero is special lol)
            # if digit not in ("0","9"):
            #     log10prob += 1

            if log10prob < log10min_prob:
                log10min_prob = log10prob
                min_prob_index = index

            # build string to print to console
            rgb_color = cmap(norm( log10prob ))[:3]
            ANSI_COLOR = rgb_to_ANSICtrl( rgb_color )
            str_array.append( ANSI_COLOR + digit )
            for _ in range(len_group-2):
                str_array.append(digit)
            str_array.append( digit + ANSI_RESET )

            index += len_group

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
        for i in range(0, (len(val)-1)//line_length + 1 ):
            index_i = i*line_length
            index_f = (i+1)*line_length
            line = ''.join(str_array[index_i:index_f])
            Console.out(f"        {line}")



        # save to table
        messages_table["const"].append(const)
        messages_table["base"].append(b)
        messages_table["exp"].append(n_qstr)
        messages_table["coeff"].append(Q_qstr)
        messages_table["index"].append(min_prob_index)
        messages_table["log10/logb * i/n"].append( ln10lnb * min_prob_index/abs(float(exponent_value)) ) # this could identify special "structure points" within the digits
        messages_table["# of digits"].append(len(str_array))
        messages_table["log10(prob)"].append(str(log10min_prob))
        messages_table["string"].append(val)

        # save table to file
        update_text("Saving digit_messages.txt")
        save_table_to_txt( "digit_messages.txt", messages_table, column_titles=messages_column_titles, column_padding=3 )

        interesting_orders_table["const"].append(const)
        interesting_orders_table["base"].append(b)
        interesting_orders_table["exp"].append(n_qstr)
        interesting_orders_table["coeff"].append(Q_qstr)
        interesting_orders_table["interesting?"].append("Yes")

        update_text("Saving orders table")
        orders_table = concatenate_tables(interesting_orders_table, uninteresting_orders_table)

        save_table_to_txt( "computed_orders.txt", orders_table, column_titles=orders_column_titles, column_padding=3 )



task(
    run,
    start_text="Computing digits",
    end_text="Computed digits",
    fail_text="Failed to compute digits",
    pass_update_text=True,
    exit_on_fail=True
)()

print("Done! :)")
# Qs = [ (1,1), (1,4), (4,3), (2,1), (1,6), (1,8), (3,8), (1,24), (3, 16), (9,16) ]
# labels = []
# for Q in Qs:
#     label = (str(Q[0]) if Q[0] != 1 else "") + "pi^2" + ("/"+str(Q[1]) if Q[1] != 1 else "")
#     labels.append(label)
#
# for b in range(9,16): #range(2,13):
#     norm = mpl.colors.LogNorm(vmin=1e-7, vmax=b**(-1))
#     cmap = subset_colormap("YlOrRd_r", new_min=0, new_max=1)
#
#     p = pi[:num_len_base10]
#     gpfpi = gmpy2.mpfr(p)
#     num_len_baseb = int( math.log(10)/math.log(b) * ( num_len_base10 + 10 ) )
#
#     print(f"base {b}: ")
#     for Q, label in zip(Qs, labels):
#         p2 = str( gpfpi**2.0 * gmpy2.mpq(*Q) )
#         n = base(p2, input_base=10, output_base=b, max_depth=num_len_baseb, string=True, recurring=False)
#
#         # colorize repeats
#         repeats = [ match.group() for match in matcher.finditer(n) ]
#         str_array = []
#
#         for repeat in repeats:
#             len_repeat = len(repeat)
#             if len_repeat == 1:
#                 str_array.append(repeat)
#                 continue
#             digit = repeat[0]
#             # rgb_color = hexcolor_to_rgb(color_conv[ len_repeat if len_repeat <= max(list(color_conv.keys())) else max(list(color_conv.keys())) ])
#
#             str_array.append( rgb_to_ANSICtrl( cmap(norm( b**(-len_repeat) ))[:3] ) + digit )
#             for _ in range(len_repeat - 2):
#                 str_array.append(digit)
#             str_array.append( digit + ANSI_RESET )
#
#         print(f"    {label} base {b}: ")
#         for i in range(0, (len(n)-1)//line_length + 1 ):
#             index_i = i*line_length
#             index_f = (i+1)*line_length
#             line = ''.join(str_array[index_i:index_f])
#             print(f"        {line}")
#
# #
# sys.exit()
#
# const_labels = ["e^pi"] #["phi","e",
# const_exponents = [ [-2,-1.5,-1,-0.5,0.5,1,1.5,2], [-2,-1.5,-1,-0.5,0.5,1,1.5,2] ] # [1, 0.5, 1.5], [-2,-1.5,-1,-0.5,0.5,1,1.5,2],
# for constant, exponents in zip(const_labels, const_exponents):
#     if constant in ["phi","e"]:
#         const_data = open(f"library/constants/{constant}.txt","rt").read()
#     else:
#         pi_data = open(f"library/constants/pi.txt","rt").read()
#         e_data = open(f"library/constants/e.txt","rt").read()
#         if constant == "e^pi":
#             pi_num = gmpy2.mpfr(pi_data[:num_len_base10])
#             e_num = gmpy2.mpfr(e_data[:num_len_base10])
#             e_pi_num = e_num**pi_num
#             const_data = str(e_pi_num)
#         elif constant == "e^pi - pi":
#             pi_num = gmpy2.mpfr(pi_data[:num_len_base10])
#             e_num = gmpy2.mpfr(e_data[:num_len_base10])
#             e_pi_pi_num = e_num**pi_num - pi_num
#             const_data = str(e_pi_pi_num)
#
#
#     labels = []
#     for exponent in exponents:
#         label = constant + ("^"+str(exponent) if exponent != 1 else "")
#         labels.append(label)
#
#     for b in range(2,23): #range(2,13):
#         norm = mpl.colors.LogNorm(vmin=1e-7, vmax=b**(-1))
#         cmap = subset_colormap("YlOrRd_r", new_min=0, new_max=1)
#
#         p = const_data[:num_len_base10]
#         gpfp = gmpy2.mpfr(p)
#         num_len_baseb = int( math.log(10)/math.log(b) * ( num_len_base10 + 10 ) )
#
#         print(f"base {b}: ")
#         for exponent, label in zip(exponents, labels):
#             p2 = str( gpfp**exponent )
#             n = base(p2, input_base=10, output_base=b, max_depth=num_len_baseb, string=True, recurring=False)
#
#             # colorize repeats
#             repeats = [ match.group() for match in matcher.finditer(n) ]
#             str_array = []
#
#             for repeat in repeats:
#                 len_repeat = len(repeat)
#                 if len_repeat == 1:
#                     str_array.append(repeat)
#                     continue
#                 digit = repeat[0]
#                 # rgb_color = hexcolor_to_rgb(color_conv[ len_repeat if len_repeat <= max(list(color_conv.keys())) else max(list(color_conv.keys())) ])
#
#                 str_array.append( rgb_to_ANSICtrl( cmap(norm( b**(-len_repeat) ))[:3] ) + digit )
#                 for _ in range(len_repeat - 2):
#                     str_array.append(digit)
#                 str_array.append( digit + ANSI_RESET )
#
#             print(f"    {label} base {b}: ")
#             for i in range(0, (len(n)-1)//line_length + 1 ):
#                 index_i = i*line_length
#                 index_f = (i+1)*line_length
#                 line = ''.join(str_array[index_i:index_f])
#                 print(f"        {line}")

# nnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
# ttttttttttttttttttttt
# ssssssssssssssssss
# rrrrrrrrrrrrrrrr
# ddddddddddddd
# DDDDDDDDDDDDD
# kkkkkkkkkkkk
# mmmmmmmmmmmm
# zzzzzzzzzzz
# bbbbbbbb
# ppppppp
# vvvvvvv
# ffffff
# ggggg
# NNNN
# TTT
# jj
# SS
# CC
# ll
# Z
