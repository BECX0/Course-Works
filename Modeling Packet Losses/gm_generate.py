#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     sgm_generate.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2020-03-25
#           2023-04-10
#
# @brief    A function for generating loss pattern based on the simple
#           Guilbert model (SGM).
#

import numpy as np
import sys


def gm_generate(len, p, q, h):
    """
    Generate a binary sequence of 0 (GOOD) and 1 (BAD) of length len
    from the SGM specified by transition probabilites 'p' (GOOD->BAD)
    and 'q' (BAD->GOOD).

    This function assumes that the SGM starts in GOOD (0) state.

    Examples:

    seq = sgm_generate(100, 0.95, 0.9)
    """

    seq = np.zeros(len)
    # check transition probabilites
    if p < 0 or p > 1:
        sys.exit("The value of the transition probability p is not valid.")
    elif q < 0 or q > 1:
        sys.exit("The value of the transition probability q is not valid.")
    elif h < 0 or h > 1:
        sys.exit("The value of the transition probability h is not valid.")
    else:
        tr = [p, q, h]

    # create a random sequence for state changes
    statechange = np.random.rand(len)
    bad_lossless = np.random.rand(len)  # bad state without packet loss
    # Assume that we start in GOOD state (0).
    state = 0
    # main loop
    for i in range(len):
        if statechange[i] <= tr[state]:
        # transition into the other state
            state ^= 1
        # add a binary value to output
        seq[i] = state
        if state == 1 and bad_lossless[i] < h:  # bad state without packet loss
            seq[i] = 0
    return seq


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--length",
        help="the length of the loss pattern to be generated; default is 10",
        default=10,
        type=int)
    parser.add_argument(
        "-p",
        help="GOOD to BAD transition probability; default is 0.95",
        default=0.95,
        type=float)
    parser.add_argument(
        "-q",
        help="BAD to GOOD transition probability; default is 0.9",
        default=0.9,
        type=float)
    parser.add_argument(
        "-h0",
        help="BAD lossless probability; default is 0.2",
        default=0.2,
        type=float)
    args = parser.parse_args()
    gm_generate(args.length, args.p, args.q, args.h0)