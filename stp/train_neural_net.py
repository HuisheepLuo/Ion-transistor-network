# This code is modified from Cross-sim example.

import sys
import os
sys.path.append(os.getcwd())

import time
import pandas as pd
import numpy as np
import pkg_resources

# define path to the stored data (neural net data and lookup tables)
datapath = pkg_resources.resource_filename("cross_sim","data")
# datapath = os.getcwd()


# import all classes used
from cross_sim import Backprop, Parameters
from cross_sim import plot_tools as PT
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

class Logger(object):
    """
    object to allow stdout to be directed to both the terminal and a file simultaneously
    """
    def __init__(self, filename=None):
        self.terminal = sys.stdout
        if filename:
            self.log = open(filename, "w",1) # the 1 makes it write after every newline, w means re-write the file (a would append)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message)
        except AttributeError:
            pass

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command
        try:
            self.log.flush()
        except AttributeError:
            pass
        self.terminal.flush()
        pass

    def change_file(self,filename):
        """
        changes the output file
        """
        try:
            self.log.close()
        except AttributeError:
            pass
        self.log = open(filename, "w",1) # the 1 makes it write after every newline, w means re-write the file (a would append)


class train_neural_net(object):
    """
    Creates a neural network training object that sets all the required parameters and runs the neural network training
    The results are saved to file

    """
    def __init__(self, outdir):
        """

        :param outdir: the directory in which to save all the simulation results
        :return:
        """
        self.outdir = outdir

        # create outdir if it does not exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        self.output_log  = Logger()
        sys.stdout = self.output_log



    def set_params(self, lookup_table=None,
                   clipscale=1.5, Gmin_relative=0.25, Gmax_relative=0.75, disable_a2d=False):
        """
        This function should be called before train and sets all parameters of the neural core simulator

        :param model: what dataset to use:  numeric lookup
        :param lookup_table: a string defining the lookup table model to use, should either be "local" or the name of a directory under data/lookup_tables
        :param clipscale: # default clipping range, 0 means no clipping
        :param Gmin_relative: min of weight range to use (works with clipscale to set scaling)
        :param Gmax_relative: max of weight range to use
        :return: params, a parameter object with all the settings

        # the lookup table defines a large conductance range, but using only part of it may give better results
        # Gmin_relative and Gmax_relative define the fraction of the lookup table (25% to 75% of the range) to target using
        # weights can go outside the targeted range, but the weight scaling will be based on the targeted range.

        """


        #######################




        ################  create parameter objects with all neural core settings for first core
        params = Parameters()

        params.xbar_params.weights.minimum = 0.1
        params.xbar_params.weights.maximum = 1

        params.algorithm_params.disable_clipping = False

        if disable_a2d:
            # set clipping parameters to give ideal accuracy
            params.algorithm_params.col_input.maximum = 100
            params.algorithm_params.col_input.minimum = -100
            params.xbar_params.col_input.bits = 0

            params.algorithm_params.row_input.maximum = 100
            params.algorithm_params.row_input.minimum = -100
            params.xbar_params.row_input.bits = 0

            params.algorithm_params.col_output.maximum = 100000
            params.algorithm_params.col_output.minimum = -100000
            params.xbar_params.col_output.bits = 0

            params.algorithm_params.row_output.maximum = 100000
            params.algorithm_params.row_output.minimum = -100000
            params.xbar_params.row_output.bits = 0

            params.algorithm_params.col_update.maximum = 100
            params.algorithm_params.col_update.minimum = -100
            params.xbar_params.col_update.bits = 0

            params.algorithm_params.row_update.maximum = 100
            params.algorithm_params.row_update.minimum = -100
            params.xbar_params.row_update.bits = 0
        else:  # use calibrated bit precision/input ranges to minimize error
            params.algorithm_params.col_input.maximum = 1
            params.algorithm_params.col_input.minimum = -1
            params.xbar_params.col_input.bits = 7
            params.xbar_params.col_input.sign_bit = True  # total bits  =  bits + sign_bit

            params.algorithm_params.row_input.maximum = 0.5*2
            params.algorithm_params.row_input.minimum = -0.5*2
            params.xbar_params.row_input.bits = 7
            params.xbar_params.row_input.sign_bit = True  # total bits  =  bits + sign_bit

            params.algorithm_params.col_output.maximum = 2*2
            params.algorithm_params.col_output.minimum = -2*2
            params.xbar_params.col_output.bits = 7
            params.xbar_params.col_output.sign_bit = True  # total bits  =  bits + sign_bit


            params.algorithm_params.row_output.maximum = 4*1.5
            params.algorithm_params.row_output.minimum = -4*1.5
            params.xbar_params.row_output.bits = 7
            params.xbar_params.row_output.sign_bit = True  # total bits  =  bits + sign_bit

            params.algorithm_params.col_update.maximum = 1
            params.algorithm_params.col_update.minimum = -1
            params.xbar_params.col_update.bits = 4
            params.xbar_params.col_update.sign_bit = True  # total bits  =  bits + sign_bit


            params.algorithm_params.row_update.maximum = 0.25*params.algorithm_params.col_output.maximum*0.1
            params.algorithm_params.row_update.minimum = -0.25*params.algorithm_params.col_output.maximum*0.1
            params.xbar_params.row_update.bits = 6
            params.xbar_params.row_update.sign_bit = True  # total bits  =  bits + sign_bit

        # set lookup table
        params.numeric_params.update_model="DG_LOOKUP"

        if  lookup_table == "local":  # lookup table stored in outdir
            params.numeric_params.dG_lookup.file_decreasing = os.path.join(self.outdir, 'dG_decreasing.txt')
            params.numeric_params.dG_lookup.file_increasing = os.path.join(self.outdir, 'dG_increasing.txt')
        elif lookup_table is None:
            pass
        else:
            lookup_dir = os.path.join(os.path.join(datapath, 'lookup_tables') , lookup_table)
            inc_file = os.path.join(lookup_dir, 'dG_increasing.txt')
            dec_file = os.path.join(lookup_dir, 'dG_decreasing.txt')

            if os.path.isfile(inc_file) and os.path.isfile(dec_file):
                params.numeric_params.dG_lookup.file_decreasing = dec_file
                params.numeric_params.dG_lookup.file_increasing = inc_file
            else:
                raise ValueError("Undefined Lookup Table Model")


        params.numeric_params.dG_lookup.Gmin_relative = Gmin_relative
        params.numeric_params.dG_lookup.Gmax_relative = Gmax_relative

        # set matrix clipping limits to clipscale (limits are further scaled in train)
        params.algorithm_params.weights.maximum = clipscale
        params.algorithm_params.weights.minimum = -clipscale

        return params


    def train(self, filename, dataset, params=None, scale_weights=True, params2=None, plot_weight_hist=False, n_epochs=40, seed = 3422323423, wtmodel = "POS"):

        """
        This function trains a neural network on 1 of 3 datasets, using the lookup_table specified by lookup_table and saves the results

        :param filename: what file to save the simulation results to
        :param dataset: what dataset to use:  small, large, cyber.  The weight ranges are scaled based on the dataset.
        :param params:  a parameters object containing all the settings for the neural core.  If none, a numeric simulation is run.
                        The weight ranges are further adjusted based on the dataset, unless scale_weights=False
        :type params: Parameters, None
        :param scale_weights: if true, weights are scaled based on dataset, if false weights are not scaled.

        :param params2: if set, it contains settings to use for the second core of the neural net.  If none, settings from the first core are copied
        :param plot_weight_hist: If true, plot a histogram of the trained weights, only works for lookup table model
        :param n_epochs: the number of epochs to run training for
        :param seed:  Random seed to use


        :return:
        """

        # set the output filename
        self.output_log.change_file(os.path.join(self.outdir, filename) )


                                                                                                                                                                                                                                                                ####################   backprop related parameters

        # settings for Ncore
        wtmodel = wtmodel  # model for handling negative weights

        # optimal initial weight scaling and learning rate
        matscale = "bengio"
        alpha = 0.1

        # counts
        niter = n_epochs  # epochs for training
        ncset = 0  # datums for classification, 0 = all
        ntset = 0  # datums for training, 0 = all


        # load appropriate dataset
        if dataset == "small":
            sizes = (64, 36, 10)
            trainfile = os.path.join(datapath, "backprop/small_digits/image_digit_small.train")
            testfile = os.path.join(datapath, "backprop/small_digits/image_digit_small.test")
        elif dataset == "large":
            sizes = (784,300,10)
            trainfile = os.path.join(datapath, "backprop/mnist/mnist.train")
            testfile = os.path.join(datapath, "backprop/mnist/mnist.test")
        elif dataset =="cyber":
            sizes = (256,512,9)
            trainfile = os.path.join(datapath, "backprop/file_types/cyber.train")
            testfile = os.path.join(datapath, "backprop/file_types/cyber.test")
        else:
            raise ValueError("Unknown dataset "+str(dataset))


        #######################


        time_start = time.time()

        # intialize backprop simulation and the two neural cores
        bp = Backprop(sizes, seed=seed)
        self.bp =bp # store backprop object to self to access analytics after training
        bp.alpha = alpha # set learning rate

        if params is None:
            model = "numeric"
            bp.ncore(which=1,wtmodel=wtmodel,truncate=0)
            bp.ncore(which=2,wtmodel=wtmodel,truncate=0)
        else:
            model = "lookup"
            clipscale = params.algorithm_params.weights.maximum  # the weights are scaled by the original setting in set_params above
            params=params.copy()
            if params2 is None:
                params2=params.copy()
            else:
                params2 = params2.copy()
            if scale_weights:
                # weight scaling different for datasets, calibrated to maximize dynamic range:
                if dataset == "small":
                    baseline_mat1 = 0.866
                    baseline_mat2 = 1.93
                elif dataset == "large":
                    baseline_mat1 = 0.218
                    baseline_mat2 = 1.05
                elif dataset =="cyber":
                    baseline_mat1 = 0.219
                    baseline_mat2 = 0.444
                else:
                    raise ValueError("Unknown dataset "+str(dataset))

                # set matrix clipping limits
                params.algorithm_params.weights.maximum*= baseline_mat1
                params.algorithm_params.weights.minimum*= baseline_mat1

                ### store weights
                # params.analytics_params.store_weights = True
                # params.analytics_params.max_storage_cycles = 1e7
                # params2.analytics_params.store_weights = True
                # params2.analytics_params.max_storage_cycles = 1e7

                ###############  Modify settings (clipping limits) for second core

                params2.algorithm_params.weights.maximum *= baseline_mat2
                params2.algorithm_params.weights.minimum *= baseline_mat2

                print("Matrix 1 Weight Limit = ",params.algorithm_params.weights.maximum, " Matrix 2 Weight Limit = ",params2.algorithm_params.weights.maximum)

            # set neural core parameters
            bp.ncore(which=1, wtmodel=wtmodel, style="new", use_params_only=True, params=params)
            bp.ncore(which=2, wtmodel=wtmodel, style="new", use_params_only=True, params=params2)


        # load the training and the test data
        if dataset=="cyber":  # need to scale cyber input data differently
            ntrain,indata_train,result_train,info = bp.read_inputs(trainfile,shuffle=1,scale="colgauss",scalereturn=1)
            ntest,indata_test,result_test = bp.read_inputs(testfile,scale="colgauss",scaleparam=info)
        else:
            ntrain,indata_train,result_train,info = bp.read_inputs(trainfile,scale="gauss",scalereturn=1)
            ntest,indata_test,result_test = bp.read_inputs(testfile,scale="gauss",scaleparam=info)


        traindata = indata_train.copy()
        trainresult = result_train.copy()
        testdata = indata_test.copy()
        testresult = result_test.copy()


        # set the initial backpropogation weights
        bp.random_weights(scale=matscale)



        # print the title of the table so that it can be interpreted by plot tools
        # title is enclosed by '#' chars
        # parameters specified by keyword1=value1 keyword2=value2

        print("\ncount, average error, max error, fraction correct\n")
        if model =="lookup":
            if params.algorithm_params.disable_clipping:
                print ("#Training model="+model+" clip=disabled"+
                       " seed="+str(seed)+
                       " #\n")
            else:
                print ("#Training model="+model+" clip="+str(clipscale)+
                       " row update bits="+str(params.xbar_params.row_update.bits)+" col update bits="+str(params.xbar_params.col_update.bits)+
                       " row input bits="+str(params.xbar_params.row_input.bits)+" col input bits="+str(params.xbar_params.col_input.bits)+
                       " row output bits="+str(params.xbar_params.row_output.bits)+" col output bits="+str(params.xbar_params.col_output.bits)+
                       " row update max="+str(params.algorithm_params.row_update.maximum)+" col update max="+str(params.algorithm_params.col_update.maximum)+
                       " row input max="+str(params.algorithm_params.row_input.maximum)+" col input max="+str(params.algorithm_params.col_input.maximum)+
                       " row output max="+str(params.algorithm_params.row_output.maximum)+" col output max="+str(params.algorithm_params.col_output.maximum)+
                       " seed="+str(seed)+
                       " #\n")
        else:
            print("#Training model="+model+
                   " seed="+str(seed)+
                   " #\n")


        # print the initial classification accuracy
        bp.ndata = ntest
        bp.indata = testdata
        bp.answers = testresult
        count, frac = bp.classify(n=ncset)
        print("%d %g %g %g" % (0, 0.0, 0.0, frac))


        # load the training data
        bp.ndata = ntrain
        bp.indata = traindata
        bp.answers = trainresult

        time1 = time.time()

        # loop over training, one iteration at a time
        # do accuracy test on test data at each iteration
        for k in range(niter):
            aveerror, maxerror = bp.train(n=ntset, debug=1)

            # load test data and classify
            bp.ndata = ntest
            bp.indata = testdata
            bp.answers = testresult
            count, frac = bp.classify(n=ncset, debug=1)

            # reload training data
            bp.ndata = ntrain
            bp.indata = traindata
            bp.answers = trainresult

            # print results
            print("%d %g %g %g" % (k + 1, aveerror, maxerror, frac))

        # print timings
        time2 = time.time()
        cpu = time2 - time1
        print("\nCPU seconds = %g\n" % cpu)

        time_stop = time.time()
        cpu = time_stop - time_start
        print("\nTotal CPU seconds = %g" % cpu)

        if plot_weight_hist==True and model=="lookup":
            ##### plot histogram of trained weights

            fig = plt.figure(figsize=(1.75,1.75))

            clip_min = bp.ncores[0].neural_core.params.xbar_params.weight_clipping.minimum
            clip_max = bp.ncores[0].neural_core.params.xbar_params.weight_clipping.maximum

            #scale matrix back into conductance range in uS
            scaling = bp.ncores[0].neural_core.params.numeric_params.dG_lookup.Gmax_clip/clip_max*1e6

            matrix = bp.ncores[0].neural_core.core.matrix * scaling
            n, bins, patches = plt.hist(matrix.flatten(), density=True, bins=50, range=(clip_min* scaling,clip_max* scaling))
            plt.xlabel(r"Conductance ($\mu$S)")
            plt.ylabel("Probability Density")
            plt.xlim([clip_min* scaling,clip_max* scaling])

            print("the weight minimum is ",bp.ncores[0].neural_core.params.xbar_params.weights.minimum)
            print("the weight maximum is ",bp.ncores[0].neural_core.params.xbar_params.weights.maximum)
            print("the weight clipping minimum is ",clip_min)
            print("the weight clipping maximum is ",clip_max)
        '''

        '''





    def plot_training(self, filenames, dataset, plot_linear=False, legend = ("Exp. Derived","Ideal Numeric"), ylim =None, plot_small=False):
        """
        Creates and saves a plot of the training results

        :param filenames: list of filenames of the data to plot
        :param dataset: what dataset to use for labels:  small, large, cyber
        :param outdir: the output directory to store figures to
        :param plot_linear: Use a linear scale or a log scale on the plot
        :param plot_small:  make smaller figures suitable for IEDM/VLSI papers
        :return:
        """

        ########## load data using plot tools (a set of tools designed to interpret the files saved by train)

        e = PT.Extract()

        titles,tables = e.table_read_all(filenames, {})

        # create emply lists to store the results for each file
        epoch  = []
        error = []
        model = []

        for i,table in enumerate(tables):
          epoch.append( PT.extract_column_table(table,1) )
          accuracy = PT.extract_column_table(table,4)
          error.append( [100.0 - 100.0*value for value in accuracy] )
          model.append(PT.extract_keyword(titles[i],"model"))


        ###########  create plot

        if not plot_small:
            fig = plt.figure(figsize=(1.75,1.75))
            ax = fig.add_subplot(111)
            mpl.rcParams['font.size'] = 8
        else:
            # plot settings of VLSI/IEDM sized figs
            fig = plt.figure(figsize=(1.1,1.1))
            ax = fig.add_subplot(111)
            mpl.rcParams['font.size'] = 6
            plt.xticks([0,10,20,30,40,50,60,70,80,90,100])

        # colors =[]
        # colors.append( "#263472")
        # colors.append ('#E9532D')




        for ind in range(len(epoch)):
            if plot_linear:
                plt.plot(epoch[ind],100-np.array(error[ind]),'-', linewidth=1,hold=True)
            else:
                plt.semilogy(epoch[ind],error[ind],'-', linewidth=1)#, color=colors[ind])

        if not plot_linear:
            ax.invert_yaxis()
            ax.set_yticklabels(['100','99','90','0'])


        # ********** define plot settings
        # mpl.rcParams['font.sans-serif']='Helvetica'
        # mpl.rcParams['font.family']='sans-serif'
        mpl.rcParams['axes.linewidth']=0.5
        mpl.rcParams['axes.titlesize']='medium'
        # ax.tick_params(which='both', direction='out', pad=1)


        if plot_linear:
            fig.subplots_adjust(left=0.22, bottom=0.17)
            plt.ylim([0,100])
        else:
            if not plot_small:
                fig.subplots_adjust(left=0.185, bottom=0.175, top=0.9, right = 0.95)
            else:
                fig.subplots_adjust(left=0.22, bottom=0.21, top=0.91, right = 0.95)

        if ylim:
            plt.ylim(ylim)

        plt.legend(legend,loc=4,frameon=False, fontsize=7)
        plt.xlabel("Training Epoch", labelpad=0.5)
        plt.ylabel("Accuracy", labelpad=0.5)

        if dataset =="small":
            plt.title("Small Digits") # 64x36x10
            plt.savefig(os.path.join(self.outdir,"small_image.png"),dpi=1200,transparent=True)
            plt.savefig(os.path.join(self.outdir,"small_image.eps"),format='eps',dpi=1200,transparent=True)
        elif dataset == "large":
            if plot_small: plt.title("MNIST", y=0.95) # 784x300x10
            else: plt.title("Large Digits") # 784x300x10
            plt.savefig(os.path.join(self.outdir,"large_image.png"),dpi=1200,transparent=True)
            plt.savefig(os.path.join(self.outdir,"large_image.eps"),format='eps',dpi=1200,transparent=True)
        elif dataset == "cyber":
            plt.title("File Types", y=0.95) #  256x512x9
            plt.savefig(os.path.join(self.outdir,"file_types.png"),dpi=1200,transparent=True)
            plt.savefig(os.path.join(self.outdir,"file_types.eps"),format='eps',dpi=1200,transparent=True)




# 4. Call main()

if __name__ == '__main__':

    epochs = 40
    # define the directory to save output to
    outdir = os.getcwd()+"/stp/output/0.01/"

    # define the lookup table to use
    lookup_table = os.getcwd()+"/stp/noisedata_Gmin/0.01/"

    # create a training object
    train_net = train_neural_net(outdir)
    params = train_net.set_params(lookup_table=lookup_table)

    # run training for each dataset

    ##### small images
    train_net.train(filename="small_images_lookup.txt", dataset="small", params=params, n_epochs=epochs)
    # train_net.plot_training([outdir+"small_images_lookup.txt"], dataset="small",plot_linear=False)

    ##### large images
    # train_net.train(filename="large_lookup.txt", dataset="large", params=params, n_epochs=epochs)
    # train_net.plot_training([outdir+"large_lookup.txt"], dataset="large",plot_linear=False)

    ##### add numeric one as comparison
    # train_net.train(filename="small_images_numeric.txt", dataset="small", params=None)
    # train_net.train(filename="small_images_lookup.txt", dataset="small", params=params, plot_weight_hist=True, n_epochs=epochs)


    # train_net.train(filename="large_images_numeric.txt", dataset="large", params=None)
    # train_net.train(filename="large_images_lookup.txt", dataset="small", params=params, plot_weight_hist=True, n_epochs=epochs)


    plt.show()



