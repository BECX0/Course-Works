import numpy as np
import sys


def calculate_sgm(data):
    seq = np.loadtxt(data)  # read data
    seq = seq.astype(int)  # change type from float to int
    len = seq.size  # calculate length of the data
    seq_str = "".join(map(str, seq))  # change data to string
    # count 1, 0, 01, 10 in the string
    n1 = seq_str.count("1")
    n0 = seq_str.count("0")
    n01 = seq_str.count("01")  # search special pattern
    n10 = seq_str.count("10")
    if n0 == 0:
        sys.exit("can not calculate sgm parameters due to no good state")
    else:
        p = n01 / n0
    if n1 == 0:
        sys.exit("can not calculate sgm parameters due to no bad state")
    else:
        q = n10 / n1
    print("model sgm p =", p, ", q =", q)
    return len, p, q

def calculate_gm(data):
    seq = np.loadtxt(data)
    seq = seq.astype(int)
    len = seq.size
    seq_str = "".join(map(str, seq))

    n1 = seq_str.count("1")
    n11 = seq_str.count("11")  # search special pattern
    n10 = seq_str.count("10")
    n111 = seq_str.count("111")
    n101 = seq_str.count("101")
    # count of pattern 111 / all 3 bit patterns
    if len <= 2:
        sys.exit("can not calculate gm parameters due to too short sequence")
    else:
        p111 = n111 / (len - 2)
        p101 = n101 / (len - 2)
    a = n1 / len
    # assuming already know 1, the prob of next symbol is still 1
    if n11 + n10 == 0:
        sys.exit("can not calculate gm parameters due to no 10 and 01 pattern")
    else:
        b = n11 / (n11 + n10)
    if p101 + p111 == 0:
        sys.exit("can not calculate gm parameters due to no 101 and 111 pattern")
    else:
        c = p111 / (p101 + p111)

    q = 1 - ((a * c - b * b) / (2 * a * c - b * (a + c)))
    h = 1 - b / (1 - q)
    p = a * q / (1 - h - a)
    print("model gm p =", p, ", q =", q, ", h =", h)
    return len, p, q, h