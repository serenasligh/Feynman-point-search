import gmpy2, math, re, sys, random, time
import numpy as np

from tools.user_tools import Console, task, subset_colormap, save_table_to_txt
from tools.label_tools import make_frac_label, make_label
from tools.math_tools import standard_notation, feynman_point_probability, generate_rationals, cartprod
from tools.baseconvert import base
from tools.ANSI_tools import color_text, ANSI_RESET, rgb_to_ANSICtrl

import matplotlib as mpl

from constants import log10prob_limit, line_length, num_len_base10, constant_values, constants_to_check, output_file

prob_library = {} # A dictionary to save the output of previous probability calculations so as to avoid exceeding python's recursion depth limit.
constants_archive = {} # Save the previous computed constants to a dictionary, so that they can be referenced for conversion into other bases.

def get_constant_value( const_key, exp, coeff, constants_archive=constants_archive, constant_values=constant_values ):
    """
    Computes the value of the constant.

    const_key   - (string) The key associated with the constant in the dictionary 'constants'.
    exp         - (2-tuple of integers) A 2-tuple representing a rational number. The first element
                  is the numerator and the second element is the denominator.
    coeff       - (2-tuple of integers) A 2-tuple representing a rational number. The first element
                  is the numerator and the second element is the denominator.

    output      - (gmpy2.mpfr) The numerical value of the constant.
    """
    if const_key not in constants_archive:
        constants_archive[const_key] = {}
    if exp not in constants_archive[const_key]:
        constants_archive[const_key][exp] = {}
    if coeff not in constants_archive[const_key][exp]:
        constants_archive[const_key][exp][coeff] = gmpy2.mpq(*coeff)*(constant_values[const_key]**gmpy2.mpq(*exp))
    return constants_archive[const_key][exp][coeff]

def colorize_digit_sequences(number_string, b, log10prob_limit):
    """
    Given a string of randomly sampled digits, colorizes recurring sequences of digits according to their probability of occurring within the first N digits.

    number_string   - (string) The number's digit expansion in any base. There is no limit to the
                      length of this string.
    b               - (int) The base of the number to be colorized.
    log10prob_limit - (int) A negative integer representing the base-10 logarithm of the minimum
                      probability to coloring. More negative integers will result in only extremely
                      improbable digit sequences being colored.

    output:
    str_array       - (list of strings) The number's digit expansion, with ANSI color characters
                      inserted. The output is represented this way so as to maintain each digit's
                      index.
    log10min_prob   - (float) A negative number representing the base-10 logarithm of the most
                      improbable sequence of digits observed in the constant.
    min_prob_index  - (int) The index of the most improbable sequence of digits in the constant.
    """

    norm = mpl.colors.Normalize(vmin=log10prob_limit-1, vmax=0)
    cmap = subset_colormap("YlOrRd_r", new_min=0, new_max=0.7)

    # Match all occurrences of digits repeating themselves twice or more successively.
    matcher = re.compile(r'(.)\1*')
    consec_num_groups = [ match.group() for match in matcher.finditer(number_string) ]

    str_array = []

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
        elif (index == 0) and (digit == "0"):
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

        index += len_group

    return str_array, log10min_prob, min_prob_index

def print_str_array(str_array, line_length, indent=6):
    number_of_lines = (len(str_array)-1)//line_length + 1
    for i in range(0, number_of_lines):
        index_i = i*line_length
        index_f = (i+1)*line_length
        line = ''.join(str_array[index_i:index_f])
        Console.out(" "*indent + f"{line}")

def run(constants_to_check, **kwargs):

    update_text = kwargs["update_text"]

    update_text("Computing constants")

    output_table_ordered_column_titles = [ "Constant", "Base", "Exp.", "Coeff.", "Minimum Log10(P)", "Index", "String" ]
    output_table = { key: [] for key in output_table_ordered_column_titles }

    i = 1
    len_constants_to_check = len(constants_to_check)
    for b, const_key, exp, coeff in constants_to_check:
        # Following code gets run for each constant to analyze.

        label = make_label(const_key,exp,coeff,b)

        update_text(f"({i}/{len_constants_to_check}) Analyzing {label}")

        ln10lnb = math.log(10)/math.log(b)
        num_len_baseb = int( ln10lnb * ( num_len_base10 + 10 ) )

        constant_value = get_constant_value( const_key, exp, coeff )
        val_str_base_10 = standard_notation( str( constant_value ) )
        val_str_base_b = base(val_str_base_10, input_base=10, output_base=b, max_depth=num_len_baseb, string=True, recurring=False) # convert number string to base b

        integer_part, fractional_part = val_str_base_b.split(".")

        fractional_part_str_array, log10min_prob, min_prob_index = colorize_digit_sequences(fractional_part, b, log10prob_limit)

        str_array = [ e for e in integer_part ] + ["."] + fractional_part_str_array

        if (log10min_prob < log10prob_limit):
            # Print colorized constant to terminal.
            Console.newline()
            Console.out(f"{label} (log10(min prob): {str(log10min_prob)}):")
            Console.newline()
            print_str_array(str_array, line_length)

            # Save data about the constant to an output txt file.
            exp_label = make_frac_label(exp, as_leading_coeff=False)
            coeff_label = make_frac_label(coeff, as_leading_coeff=False)

            output_table_row = (const_key, b, exp_label, coeff_label, log10min_prob, min_prob_index, val_str_base_b)
            for column_title, row_item in zip(output_table_ordered_column_titles, output_table_row):
                output_table[column_title].append( row_item )

            update_text(f"Saving data to {output_file}")
            save_table_to_txt(output_file, output_table, column_titles=output_table_ordered_column_titles)

        i+=1

task(
    run,
    start_text="Computing digits",
    end_text="Computed digits",
    fail_text="Failed to compute digits",
    pass_update_text=True
)(constants_to_check)
