# This code is modified from Cross-sim example.

import os
import sys
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
from mpldatacursor import datacursor
from cross_sim import process_data

if __name__ == '__main__':

    # ********* specify csv locations and properties
    # experimental measurements should be taken as alternating series of increasing and decreasing pulses.
    # Each run of set or reset pulses should be a column in set.csv or reset.csv respectively
    # the output files dG_decreasing.txt and dG_increasing.txt are the lookup tables needed for CrossSim
    folder = 'stp/noisedata_Gmin/0.01'
    set_file = 'set.csv'
    reset_file = 'reset.csv'
    header_lines = 1
    skip_footer = 0
    read_voltage = 1  # It is assumed that set/reset.csv are current values and so it needs to be scaled by the read voltage to get a conductance
    scale = 1e3 # scaling factor for plots
    scale_text = "m"  # "$\mu$" ##scaling factor label for plot labels
    extrapolate = True # if true, extrapolate dG values to G values that were not measured, if false, use the closest measured value
    nbins=50  # number of G bins to quantize lookup table into

    # create data processing object
    pd = process_data()

    # set location of csv files and data output location
    pd.datapath = folder

    # Get two lists of ndarrays, 1st of increasing pulses, 2nd of decreasing pulses.  The data in each list should be conductance values after a pulse
    # alternate data processing functions can be used for new data sources
    increasing_pulses, decreasing_pulses = pd.load_set_reset_csv(set_file=set_file, reset_file=reset_file,
                                                                 skip_header=header_lines, skip_footer=skip_footer,
                                                                 read_voltage=read_voltage)

    #######  find min/max conductance
    Gmin1, Gmax1 = pd.find_min_max(increasing_pulses)
    Gmin2, Gmax2 = pd.find_min_max(decreasing_pulses)

    Gmin = max(Gmin1,
               Gmin2)  # use max of min so that we don't get stuck in limit where a G value exists only only the increasing or only decreasing pulses
    Gmax = min(Gmax1, Gmax2)

    print("The Min G is: " + str(Gmin * scale) + "(" + scale_text + "S) The max G is " + str(
        Gmax * scale) + " (" + scale_text + "S)")


    # ******************** increasing pulses
    # bin the data and create G vs delta G plot
    binned_dG, binned_CDF, Gbins = pd.bin_data(increasing_pulses, Gmin, Gmax, nbins=nbins, fig_prefix="increasing",
                                               extrapolate=extrapolate)

    # # save lookup table
    Gbins, CDF_points, dG_array = pd.create_lookup_table(binned_dG, binned_CDF, Gbins, n_CDF_points=501,
                                                         n_CDF_end_points=0, filename="dG_increasing.txt",
                                                         n_G_points=501, filter_CDF=0, fig_prefix="increasing")



    # ******************* decreasing pulses
    # bin the data and create G vs delta G plot
    binned_dG, binned_CDF, Gbins = pd.bin_data(decreasing_pulses, Gmin, Gmax, nbins=nbins, fig_prefix="decreasing",
                                               extrapolate=extrapolate)

    # # save lookup table
    Gbins, CDF_points, dG_array = pd.create_lookup_table(binned_dG, binned_CDF, Gbins, n_CDF_points=501,
                                                         n_CDF_end_points=0, filename="dG_decreasing.txt",
                                                         n_G_points=501, filter_CDF=0, fig_prefix="decreasing")

    datacursor()
    plt.show()
