"""
PURPOSE:
Build Multilayer Perceptron Artificial Neural Networks implemented in TensorFlow (TF)

INPUT:
    REQUIRED:
    -x          File with feature information
    -y          File with values you want to predict 
    -y_name     Name of column in y with the value you want to predict (Default = Y)
    -ho         File with holdout set
    -save       Name to include in RESULTS file (i.e. what dataset are you running)
    
    OPTIONAL:
    # Input/Output Functions
        -f          Select function to perform (gs, run, full*) *Default
        -feat       File with list of features from -x to include
        -norm       T/F Normalize Y (default = T)
        -tag        Identifier string to add to RESULTS file
        -sep        Specify seperator in -x and -y (Default = '\t')
        -s_weights  T/F Save the trained weights from the trained network (only if hidden layers <= 3)
        -s_losses   T/F Save the training, validation, and testing losses from final model training

    # Specify hyperparameters (only if -f run)
        -params    Output from -f gs (i.e. SAVE_GridSearch.txt)
        -actfun     Activation function. (relu, sigmoid*) *Default
        -lrate      Learning rate. Default = 0.01
        -dropout    Dropout rate. Default = 0.1 (i.e. drop out 10% of nodes each epoch)
        -l2         Shrinkage parameter for L2 regularization. Default = 0
        -arch       MLP architecture as comma separated layer sizes (e.g. 100,50 or 200,100,50)
    
    # Training behavior 
        -max_epoch     Max number of epochs to iterate through. Default = 50,000
        -epoch_thresh  Threshold for percent change in MSE for early stopping. Default = 0.001
        -burnin        Number of epochs before start counting for early stopping. Default = 100
        -val_perc      What percent of the training set to hold back for validation. Default = 0.1
        -loss_type     Loss function to minimize during training. Only MSE available now. Default = mse

OUTPUT:
    -SAVE_GridSearch.txt    Results from grid search. Appends to SAVE_GridSearch.txt if already exists
    -RESULTS.txt            Summary of results from final model (from -f run/full)
    -SAVE_losses.csv        Training, validation, and testing losses for each epoch in final model (-s_losses t)
    -SAVE_Weights_X.csv     Final trained ANN weights for hidden layers (_HL#) and final connection (_fin)
    

EXAMPLE ON HPCC:
$ source /mnt/home/azodichr/python3-tfcpu/bin/activate
$ python ANN_mlp.py -f full -x geno.csv -y pheno.csv -y_name HT -sep ',' -ho holdout.txt -save mlp_HT -gs t -gs_reps 10 -weights xavier -norm t

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import timeit
import ANN_Functions as ANN

start_time = timeit.default_timer()
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def main():

    #####################
    ### Default Input ###
    #####################

    FUNCTION = 'full'
    
    # Input and Output info
    TAG, FEAT, SAVE, y_name, norm = '', '', 'test', 'Y', 't'
    save_weights = save_losses = 'f'

    # Hyperparameters
    actfun, arc, lrate, dropout, l2 = 'sigmoid', '10,5', 0.01, 0.1, 0.0
    params = ''

    # Grid Search Hyperparameter Space
    gs_reps = 10
    list_dropout = list_l2 = [0.0, 0.1, 0.5]
    list_lrate = [0.01, 0.001]
    list_actfun = ["sigmoid", "relu"]
    list_arch = ["10", "50", "100", "10,5", "50,25", "100,50", "10,5,5", "50,25,10", "100,50,25"]

    # Training Parameters
    max_epochs = 50000
    epoch_thresh = 0.001
    burnin = 100
    loss_type = 'mse'
    val_perc = 0.1 

    # Weight initialization
    WEIGHTS = 'random'
    mu, sigma = 0, 0.01


    ##################
    ### User Input ###
    ##################
    for i in range (1,len(sys.argv),2):
      if sys.argv[i].lower() == "-f":
        FUNCTION = sys.argv[i+1]
      if sys.argv[i].lower() == "-x":
        X_file = sys.argv[i+1]
      if sys.argv[i].lower() == "-y":
        Y_file = sys.argv[i+1]
      if sys.argv[i].lower() == '-ho':
        ho = sys.argv[i+1]
      if sys.argv[i].lower() == '-val_perc':
        val_perc = float(sys.argv[i+1])
      if sys.argv[i].lower() == '-sep':
        SEP = sys.argv[i+1]
      if sys.argv[i].lower() == '-norm':
        norm = sys.argv[i+1]
      if sys.argv[i].lower() == "-feat":
        FEAT = sys.argv[i+1]
      if sys.argv[i].lower() == "-weights":
        WEIGHTS = sys.argv[i+1]
      if sys.argv[i].lower() == "-mu":
        mu = float(sys.argv[i+1])
      if sys.argv[i].lower() == "-sigma":
        sigma = float(sys.argv[i+1])
      if sys.argv[i].lower() == "-tag":
        TAG = sys.argv[i+1]
      if sys.argv[i].lower() == "-y_name":
        y_name = sys.argv[i+1]
      if sys.argv[i].lower() == "-save":
        SAVE = sys.argv[i+1]
      if sys.argv[i].lower() == "-actfun":
        actfun = sys.argv[i+1] 
      if sys.argv[i].lower() == "-epoch_thresh":
        epoch_thresh = float(sys.argv[i+1])
      if sys.argv[i].lower() == "-epoch_max":
        epoch_thresh = int(sys.argv[i+1])
      if sys.argv[i].lower() == "-loss_type":
        loss_type = sys.argv[i+1]        
      if sys.argv[i].lower() == "-params":
        params = sys.argv[i+1]
      if sys.argv[i].lower() == "-burnin":
        burnin = int(sys.argv[i+1])
      if sys.argv[i].lower() == "-lrate":
        lrate = float(sys.argv[i+1])
      if sys.argv[i].lower() == "-l2":
        l2 = float(sys.argv[i+1])
      if sys.argv[i].lower() == "-dropout":
        dropout = float(sys.argv[i+1])
      if sys.argv[i].lower() == "-arch":
        arc = sys.argv[i+1]
      if sys.argv[i].lower() == "-s_weights":
        save_weights = sys.argv[i+1]
      if sys.argv[i].lower() == "-s_losses":
        save_losses = sys.argv[i+1]
      if sys.argv[i].lower() == "-gs_reps":
        gs_reps = int(sys.argv[i+1])



    ################
    ### Features: read in file, keep only those in FEAT if given, and define feature_cols for DNNReg.
    ################
    x = pd.read_csv(X_file, sep=SEP, index_col = 0)
    if FEAT != '':
        with open(FEAT) as f:
            features = f.read().strip().splitlines()
        x = x.loc[:,features]
    feat_list = list(x.columns)
    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in feat_list]

    print("\n\nTotal number of instances: %s" % (str(x.shape[0])))
    print("\nNumber of features used: %s" % (str(x.shape[1])))

    ################
    ### Y: read in file, keep only column to predict, normalize if needed, and merge with features
    ################
    y = pd.read_csv(Y_file, sep=SEP, index_col = 0)
    if y_name != 'pass':
        print('Building model to predict: %s' % str(y_name))
        y = y[[y_name]]
    if norm == 't':
        print('Normalizing Y...')
        mean = y.mean(axis=0)
        std = y.std(axis=0)
        y = (y - mean) / std

    df = pd.merge(y, x, left_index=True, right_index=True)
    yhat = df[y_name]

    print('\nSnapshot of data being used:')
    print(df.head())

    ################
    ### Holdout: Drop holdout set as it will not be used during grid search
    ################
    print('Removing holdout instances to apply model on later...')

    with open(ho) as ho_file:
        ho_instances = ho_file.read().splitlines()
        num_ho = len(ho_instances)
    try:
        test = df.loc[ho_instances, :]
        train = df.drop(ho_instances)
    except:
        ho_instances = [int(x) for x in ho_instances]
        test = df.loc[ho_instances, :]
        train = df.drop(ho_instances)

    val_set_index = np.random.rand(len(train)) < val_perc
    valid = train[val_set_index]
    train = train[~val_set_index]

    X_train = train.drop(y_name, axis=1).values
    X_valid = valid.drop(y_name, axis=1).values
    X_test = test.drop(y_name, axis=1).values
    Y_train = train.loc[:, y_name].values
    Y_valid = valid.loc[:, y_name].values
    Y_test = test.loc[:, y_name].values

    n_input = X_train.shape[1]
    n_samples = X_train.shape[0]
    n_classes = 1

    # Define TF Graph Placeholders 
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, 1])
    dropout_rate = tf.placeholder(tf.float32) # For dropout, allows it to be turned on during training and off during validation


    ################
    ### Grid Search: Using train:validate splits
    ################

    if FUNCTION == 'gs' or FUNCTION == 'full':
        print('Starting Grid Search...')
        gs_results = pd.DataFrame()
        gs_count = 0
        gs_length = len(list_dropout) * len(list_l2) * len(list_lrate) * len(list_actfun) * len(list_arch) * gs_reps
        for r in range(0,gs_reps):
            print(range(0, gs_reps))
            for dropout in list_dropout:
                for l2 in list_l2:
                    for lrate in list_lrate:
                        for actfun in list_actfun:
                            for arc in list_arch:
                                if gs_count % 10 == 0:
                                    print('Grid Search Status: %i out of %i' % (gs_count, gs_length))
                                
                                # Construct ANN model
                                archit, layer_number = ANN.fun.define_architecture(arc)
                                weights, biases = ANN.fun.initialize_starting_weights(WEIGHTS, n_input, n_classes, archit, layer_number, df, mu, sigma)
                                pred = ANN.fun.multilayer_perceptron(x, weights, biases, layer_number, actfun, dropout, dropout_rate)
                                loss = ANN.fun.define_loss(loss_type, y, pred, l2, weights)
                                optimizer = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss)
                                correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
                                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                                # Launch the graph
                                sess = tf.Session()
                                init = tf.global_variables_initializer()
                                sess.run(init)

                                ## Train the ANN model ##
                                epoch_count = stop_count = 0
                                old_c = 1
                                train='yes'

                                while train == 'yes':
                                    epoch_count += 1
                                    _, c = sess.run([optimizer, loss],feed_dict = {x:X_train, y:pd.DataFrame(Y_train), dropout_rate:dropout})
                                    valid_c = sess.run(loss, feed_dict = {x:X_valid, y:pd.DataFrame(Y_valid), dropout_rate:1})
                                    
                                    if epoch_count >= burnin:
                                        pchange = (old_c-valid_c)/old_c
                                        if abs(pchange) < epoch_thresh:
                                            stop_count += 1
                                            if stop_count >= 10:
                                                train='no'
                                    old_c = valid_c
                                    if epoch_count == max_epochs:
                                        train='no'

                                pred_c = sess.run(pred, feed_dict = {x:X_valid, y:pd.DataFrame(Y_valid), dropout_rate:1})
                                val_cor = np.corrcoef(Y_valid, pred_c[:,0])
                                gs_results = gs_results.append({'ActFun':actfun, 'Arch':arc, 'dropout':dropout, 'L2':l2, 'LearnRate':lrate,'Epochs':epoch_count, 'Train_Loss':c, 'Valid_Loss':valid_c, 'Valid_PCC':val_cor[0,1]}, ignore_index=True)
                                gs_count += 1
            
        if not os.path.isfile(SAVE + "_GridSearch.txt"):
            gs_results.to_csv(SAVE + "_GridSearch.txt", header='column_names', sep='\t')
        else: 
            gs_results.to_csv(SAVE + "_GridSearch.txt", mode='a', header=False, sep='\t')

        print('\n\n Grid Search results saved to: %s_GridSearch.txt\n' % SAVE)


    
    ################
    ### Run final model 
    ################

    if FUNCTION == 'full' or FUNCTION == 'run':
        
        # Grab parameters from grid search results
        if FUNCTION == 'full' or params != '':
            if FUNCTION == 'full':
                gs_res = gs_results
            
            if params != '':
                gs_res = pd.read_csv(params, sep='\t')
            
            gs_ave = gs_res.groupby(['ActFun','dropout','L2','Arch','LearnRate']).agg({
                'Valid_Loss': 'median', 'Train_Loss': 'median', 'Valid_PCC': 'median', 'Epochs': 'mean'}).reset_index()
            gs_ave.columns = ['ActFun','dropout','L2','Arch','LRate', 'VLoss_med', 'TLoss_med', 'VPCC_med', 'Epochs_mean']
            results_sorted = gs_ave.sort_values(by='VPCC_med', ascending=False)
            print('\nSnapshot of grid search results:')
            print(results_sorted.head())

            actfun = results_sorted['ActFun'].iloc[0]
            dropout = float(results_sorted['dropout'].iloc[0])
            l2 = float(results_sorted['L2'].iloc[0])
            lrate = float(results_sorted['LRate'].iloc[0])
            arc = results_sorted['Arch'].iloc[0]


        print("\n\n##########\nBuilding MLP with the following parameters:\n")
        print('Architecture: %s' % arc)
        print('Regularization: dropout = %f  L2 = %f' % (dropout, l2))
        print('Learning rate: %f' % lrate)
        print('Activation Function: %s\n\n\n' % actfun)


        # Construct ANN model
        archit, layer_number = ANN.fun.define_architecture(arc)
        weights, biases = ANN.fun.initialize_starting_weights(WEIGHTS, n_input, n_classes, archit, layer_number, df, mu, sigma)
        pred = ANN.fun.multilayer_perceptron(x, weights, biases, layer_number, actfun, dropout, dropout_rate)
        loss = ANN.fun.define_loss(loss_type, y, pred, l2, weights)
        optimizer = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Launch the graph
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        tvars = tf.trainable_variables() # Define variables to hold weights
        tvars_vals = sess.run(tvars)

        ## Train the ANN model ##
        epoch_count = stop_count = 0
        old_c = 1
        train, losses = 'yes', []

        while train == 'yes':
            epoch_count += 1
            _, c = sess.run([optimizer, loss],feed_dict = {x:X_train, y:pd.DataFrame(Y_train), dropout_rate:dropout})
            valid_c = sess.run(loss, feed_dict = {x:X_valid, y:pd.DataFrame(Y_valid), dropout_rate:1})
            test_c = sess.run(loss, feed_dict={x: X_test, y:pd.DataFrame(Y_test), dropout_rate:1})
            losses.append([epoch_count, c, valid_c, test_c])
            
            pchange = (old_c-valid_c)/old_c
            if epoch_count >= burnin:
                if abs(pchange) < epoch_thresh:
                    stop_count += 1
                    print('Early stopping after %i more below threshold' % (10-stop_count))
                    if stop_count >= 10:
                        train='no'

            if (epoch_count) % 50 == 0:
                print("Epoch:", '%i' % (epoch_count), "; Training MSE=", "{:.3f}".format(c), "; Valid MSE=", "{:.3f}".format(valid_c), '; Percent change=', str(pchange))
 
            old_c = valid_c
            if epoch_count == max_epochs or train=='no':
                print('Final MSE after %i epochs for training: %.5f and testing: %.5f' % (epoch_count, c, valid_c))
                train = 'no'

        # Predict test set and add to yhat output
        y_pred = sess.run(pred, feed_dict={x: X_test, dropout_rate:1})
        print('Predicted Y values:')
        print(y_pred[:,0])
        ho_cor = np.corrcoef(Y_test, y_pred[:,0])
        print('Holdout correlation coef (r): %.5f' % ho_cor[0,1])



        ##### Optional Outputs ####

        if save_weights == 't':
            ANN.fun.save_trained_weights(SAVE, tvars, tvars_vals, archit, feat_list)

        if save_losses == 't':
            losses_df = pd.DataFrame(losses, columns=['epoch', 'MSE_train', 'MSE_valid', 'MSE_test'])        
            losses_df.to_csv(SAVE+'_losses.csv', index=False)
        
        run_time = timeit.default_timer() - start_time
        if not os.path.isfile('RESULTS.txt'):
            out1 = open('RESULTS.txt', 'w')
            out1.write('DateTime\tRunTime\tTag\tDFs\tDFy\tTrait\tFeatSel\tWeights\tNumFeat\tHoldout\tNumHidLay\tArchit\tActFun\tEpochs\tdropout\tL2\tLearnRate\tMSE_Train\tMSE_Valid\tMSE_test\tPCC_test\n')
            out1.close()
             
        out2 = open('RESULTS.txt', 'a')
        out2.write('%s\t%0.5f\t%s\t%s\t%s\t%s\t%s\t%s\t%i\t%s\t%i\t%s\t%s\t%i\t%f\t%f\t%f\t%0.5f\t%0.5f\t%0.5f\t%0.5f\n' % (
            timestamp, run_time, TAG, X_file, Y_file, y_name, FEAT, WEIGHTS, x.shape[1], ho, layer_number, str(arc), actfun, epoch_count, dropout, l2, lrate, c, valid_c, test_c, ho_cor[0,1]))
        out2.close()
        print('\nfinished!')


if __name__ == '__main__':
    main()