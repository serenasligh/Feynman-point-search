# Feynman-point-search

This project was made to search for strings of successively repeated digits in the base-n digit expansions of irrational numbers.

Inspired by this Wikipedia article:
https://en.wikipedia.org/wiki/Six_nines_in_pi

For an as of yet unknown reason, the base-10 digits of pi contain a sequence of six nines starting at the 762nd decimal place. Assuming that pi is a 'normal' number (meaning each digit 0-9 has an equal probability of occurring at any position within the digits of pi), then we should expect to eventually find every possible sequence of numbers somewhere in the infinite digits of pi, though it seems rather unlikely that six nines might occur in succession so closely to the first digit.

## Computing the probability of the Feynman point's occurrence

The following was essentially transcribed from problem \#18 of https://www.madandmoonly.com/doctormatt/mathematics/dice1.pdf by Matthew M. Conroy.

The probability of the six nines occurring within the first 768 digits can be computed using a recursion relation:

Let *p_n* be the probability of *k* successive nines occurring within the first *n* digits, and let *b* be the probability of any digit of pi being nine (so *b = 1/10* in base-10, assuming pi is normal).

If n<k, then *p_n = 0* since there aren't enough digits for a run of *k* nines to even occur.

There are two ways that a sequence of exactly *k* nines can occur with the first n+1 digits:
- (a) *k* nines occur successively within the first *n* digits.
- (b) The final *k* digits are all nines.

The probability of (a) occurring *p_a = p_n*, according to the definition of *p_n*.

To compute the probability of (b) occurring *p_b*, we must first note that three independent events have to happen for (b) to occur:

- (1) There is *not* a run of k nines within the first *n-k* digits. This occurs with probability *p_1 = 1-p_(n-k)*.
- (2) The digit with index *n-k+1* must *not* be nine. Otherwise *k* successive nines will have occurred in the first *n* digits. This occurs with probability *p_2 = 1-b*.
- (3) The final *k* digits are all nines, occurring with probability *p_3 = b^k*.

Thus *p_b = p_1 * p_2 * p_3*. This leads us to a linear recursion relation for *p_(n+1)*:

*p_(n+1) = p_n + (1-p_(n-k)) * (1-b) * b^k*

This recursion relation is subject to a few boundary conditions:

- (1) *p_r = b^r* (e.g. If we're looking for a sequence of six nines in a random six digit number, then every digit must be a nine.)
- (2) *p_(n<r) = 0*  --> *p_(r-1) = 0*, *p_(r-2) = 0*, *...* , *p_0 = 0*

To determine the value of *p_n* for any *n*, we can start by first computing *p_k*, *p_(k+1)*, *p_(k+2)*, and so on until we reach *p_n*.

## So... exactly how improbable is the Feynman point? Are there others?

Setting *b = 1/10*, *k = 6*, and *n = 767* (since the sequence of nines terminates at digit \#767 relative to the decimal point) we find that *p = 10^-3.164 ~ 7/10,000*.

One way you could interpret this is that 7 out of every 10000 normal irrational numbers will have a sequence of nines this close to the decimal point. There are other irrational constants with even more improbable sequences of digits.

- *2 * pi = 6.2831853071* has a sequence of **seven** 9s in base-10 starting at digit \#760, with probability *p = 10^-4.164 ~ 7/100,000*.

- *(pi)^(-1/2) = 0.622A332988A* in base-11 has a sequence of six As starting at digit \#904, with *p = 10^-3.333 ~ 5/10,000*.

- Furthermore, *1/(84^2) * (pi)^(-1/2) = 0.00107857971* in base-11 has a sequence of ***NINE*** 1s starting at digit \#904, with *p = 10^-6.457 ~ 3/10,000,000*.

- *(1/2) * ln(3)^(-1/2) = 0.954064582000001399* in base-10 has a sequence of five 0s starting at digit \#14, with *p = 10^-5.041 ~ 1/100,000*.

- *(2/3) * 7^(-1/3) = 0.1332401442030* in base-5 has a sequence of eight 1s starting at digit \#33, with *p = 10^-4.154 ~ 7/100,000*.

# How to use this project

The file constants.py contains all the adjustable parameters you'll need and examples that demonstrate how to form a list of instructions ('constants_to_check') that feynman_point_search.py will use to generate constants.

Once you've configured constants.py, navigate to the folder \\Feynman-point-search\\ and run

python feynman_point_search.py

Your results will print to the terminal and save automatically to the folder \\output\\.
