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

def make_label(const,exp,coeff,b):
    coeff_label = make_frac_label(coeff)
    exp_label = make_exp_label(exp)
    label = f"{coeff_label} {const}{exp_label}, base-{b}"
    return label
