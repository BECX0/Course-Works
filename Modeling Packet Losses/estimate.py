import numpy as np
import re
import matplotlib.pyplot as plt
from calculate_parameters import calculate_gm, calculate_sgm
from gm_generate import gm_generate


def calculate_run_lengths(data):
    if type(data) == np.ndarray:  # the format of generated sequence
        seq = data
    else:  # the format of original sequence
        seq = np.loadtxt(data)
    seq = seq.astype(int)
    seq_str = "".join(map(str, seq))
    zeros = re.findall(r'0+', seq_str)  # search for 0, 00, 000...
    zeros_dir = {}  # save the occurrence numbers of run-lengths
    for zero in zeros:  # count occurrence numbers
        length = len(zero)
        zeros_dir[length] = zeros_dir.get(length, 0) + 1
    sorted_zeros = sorted(zeros_dir.items(), key=lambda x: x[0])  # reorder by run length

    ones = re.findall(r'1+', seq_str)
    ones_dir = {}
    for one in ones:
        length = len(one)
        ones_dir[length] = ones_dir.get(length, 0) + 1
    sorted_ones = sorted(ones_dir.items(), key=lambda x: x[0])
    return sorted_zeros, sorted_ones


def plot_sequences(orig_seq, gen_seq, model):
    (orig_zeros, orig_ones) = calculate_run_lengths(orig_seq)
    (gen_zeros, gen_ones) = calculate_run_lengths(gen_seq)

    keys_0s = [item[0] for item in orig_zeros]  # get run-lengths
    values_0s = [item[1] for item in orig_zeros]
    keys_1s = [item[0] for item in orig_ones]
    values_1s = [item[1] for item in orig_ones]

    gen_keys_0s = [item[0] for item in gen_zeros]
    gen_values_0s = [item[1] for item in gen_zeros]
    gen_keys_1s = [item[0] for item in gen_ones]
    gen_values_1s = [item[1] for item in gen_ones]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar([x - 0.2 for x in keys_0s], values_0s, width=0.4, label='Orig_zeros')
    ax2.bar([x - 0.2 for x in keys_1s], values_1s, width=0.4, label='Orig_ones')
    ax1.bar([x + 0.2 for x in gen_keys_0s], gen_values_0s, width=0.4, label='Gen_zeros')
    ax2.bar([x + 0.2 for x in gen_keys_1s], gen_values_1s, width=0.4, label='Gen_ones')

    ax1.set_title('Histogram of Counts using ' + model + ' Model')
    ax1.set_xlabel('Length of Zeros')
    ax1.set_ylabel('Frequency')
    ax2.set_title('Histogram of Counts using ' + model + ' Model')
    ax2.set_xlabel('Length of Ones')
    ax2.set_ylabel('Frequency')

    ax1.set_xlim(0, 16)
    ax2.set_xlim(0, 16)
    ax1.legend()
    ax2.legend()
    fig.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.2)  # adjust spacing
    plt.suptitle(orig_seq)
    plt.show()

    orig_seq_int = np.loadtxt(orig_seq)
    orig_seq_int = orig_seq_int.astype(int)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    window = 256
    [Pxx1, f1] = ax1.psd(orig_seq_int,
                         NFFT=window,
                         Fs=10,
                         detrend='mean',
                         window=np.hanning(window),
                         noverlap=int(window * 3 / 4),
                         sides='onesided')

    [Pxx2, f2] = ax2.psd(gen_seq,
                         NFFT=window,
                         Fs=10,
                         detrend='mean',
                         window=np.hanning(window),
                         noverlap=int(window * 3 / 4),
                         sides='onesided')
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax1.set_title('PSD of Original Sequence using ' + model + ' Model')
    ax2.set_title('PSD of Generated Sequence using ' + model + ' Model')
    fig.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.2)
    plt.suptitle(orig_seq)
    plt.show()


def main():
    orig_seq = "dataset-A-adsl1-cbr6.0-20090628-223500.bitmap"
    print(orig_seq)
    (len, p, q) = calculate_sgm(orig_seq)  # calculate parameters
    gen_seq = gm_generate(len, p, q, 0)  # generate new sequences with gm model
    plot_sequences(orig_seq, gen_seq, "sgm")  # plot histograms and PSD figures

    orig_seq = "dataset-A-adsl1-cbr6.0-20090628-223500.bitmap"
    (len, p, q, h) = calculate_gm(orig_seq)
    gen_seq = gm_generate(len, p, q, h)
    plot_sequences(orig_seq, gen_seq, "gm")

    orig_seq = "dataset-A-adsl5-cbr6.0-20091011-203200.bitmap"
    print(orig_seq)
    (len, p, q) = calculate_sgm(orig_seq)
    gen_seq = gm_generate(len, p, q, 0)
    plot_sequences(orig_seq, gen_seq, "sgm")

    orig_seq = "dataset-A-adsl5-cbr6.0-20091011-203200.bitmap"
    (len, p, q, h) = calculate_gm(orig_seq)
    gen_seq = gm_generate(len, p, q, h)
    plot_sequences(orig_seq, gen_seq, "gm")


if __name__ == "__main__":
    main()
