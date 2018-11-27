"""

PURPOSE: Run Parameter Sweep for Convolutional Neural Network Regressor using Tensorflow 

cnn_a.py : Grid search through parameter space
cnn_b.py : Build CNN using provided parameters

INPUTS:

  REQUIRED:
    -x      Option 1: File with input data. Images should be processed so all pixle data is in one line:
              Example:  Original        Input
                  Pic1  Pix1 Pix2 Pix3           Pix1 Pix2 Pix3 Pix4 Pix5 Pix6
                        Pix4 Pix5 Pix6      Pic1  0    0    1    1    0    1
                  Pic2  Pix1 Pix2 Pix3      Pic2  1    0    0    1    0    0
                        Pix4 Pix5 Pix6
            Option 2: Directory with image files. Will resize to fit -shape given. Note: Saves a copy of the 
                processed data to the image dir, if re-processing is needed, delete 'dir/X_processed.csv'.
    -y      File with dependent variable to predict.
    -ho     With with list of testing instances to holdout from training. (Generate using ML_Pipeline/holdout.py)
    -save   Prefix for grid search output file - note make unique for each pred problem.
    -shape  Dimensions of image: row,col. For the above sample -shape 2,3
    

  OPTIONAL:
    -f       Select function to perform (gs, run, full*) *Default
    -y_name  Name of column from -y to use if more than one column present
    -norm    T/F Normalize Y (default = T)
    -sep     Specify seperator in -x and -y (Default = '\t')
    -actfun  Activation function. Default = relu, suggested GridSearch: [relu, sigmoid]
    -lrate   Value for learning rate (L2). Default = 0.01, suggested GridSearch: [0.001, 0.01, 0.1]
    -dropout      Value for dropout regularization (dropout). Default = 0.25, suggested GridSearch: [0.0, 0.1, 0.25, 0.5]
    -l2      Value for shrinkage regularization (L2). Default = 0.1, suggested GridSearch: [0.0, 0.1, 0.25, 0.5]
    -conv_shape   Dimensions of convolutions: row,col. Default = 5,5
    -feat    List of columns in -x to use. Can also be used to re-order columns in -x
    -max_epoch  Max number of epochs to iterate through
    -epoch_thresh**  Threshold for percent change in MSE before training stops. Default: 0.001 
    -s_losses   T/F Save the training, validation, and testing losses from final model training
    -s_yhat     T/F Apply trained model to all data and save output 

** The number of training epochs (i.e. iterations) is dynamic, based on the -epoch_threshold. 
   After an initial burnin period (100 epochs here), every time the abs(% change in MSE) for the 
   validation set is below the epoch_threshold. After 10 epochs with a %change below the threshold
   training stops and the final training and validation MSE are reported


Example:

source /mnt/home/azodichr/python3-tfcpu/bin/activate

python ANN_cnn.py -f gs -x geno.csv -y pheno.csv -ho holdout.txt -feat rice_YLD_RF_1_2000.txt -y_name YLD -sep ',' -save test -norm t -gs_reps 10
Roughly based off: https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import timeit
from scipy.stats.stats import pearsonr 
import ANN_Functions as ANN

tf.logging.set_verbosity(tf.logging.INFO)
start_time = timeit.default_timer()


def main():

    #####################
    ### Default Input ###
    #####################

    FUNCTION = 'full'
    
    # Input and Output info
    SEP, TAG, FEAT, SAVE, y_name, norm = '\t', '', '', 'test', 'Y', 't'
    save_weights = save_losses = save_yhat = 'f'

    # Hyperparameters
    actfun, lrate, dropout, l2 = 'sigmoid', 0.01, 0.25, 0.0
    params = ''

    # Grid Search Hyperparameter Space
    gs_reps = 10
    list_dropout = list_l2 = [0.0, 0.1, 0.25, 0.5]
    list_lrate = [0.01, 0.001]
    list_actfun = ["sigmoid", "relu"]

    # Training Parameters
    max_epochs = 50000
    epoch_thresh = 0.001
    burnin = 10
    loss_type = 'mse'
    val_perc = 0.1 

    # Default CNN structure
    conv_r, conv_c = 5, 5
    shape_r, shape_c = int(50), int(40)

    for i in range (1,len(sys.argv),2):
      if sys.argv[i].lower() == "-x":
        X_file = sys.argv[i+1]
      if sys.argv[i].lower() == "-y":
        Y_file = sys.argv[i+1]
      if sys.argv[i].lower() == '-ho':
        ho = sys.argv[i+1]
      if sys.argv[i].lower() == '-sep':
        SEP = sys.argv[i+1]
      if sys.argv[i].lower() == "-feat":
        FEAT = sys.argv[i+1]
      if sys.argv[i].lower() == "-y_name":
        y_name = sys.argv[i+1]
      if sys.argv[i].lower() == "-norm":
        norm = sys.argv[i+1]
      if sys.argv[i].lower() == "-save":
        SAVE = sys.argv[i+1]
      if sys.argv[i].lower() == "-actfun":
        actfun = sys.argv[i+1] 
      if sys.argv[i].lower() == "-loss_type":
        loss_type = sys.argv[i+1] 
      if sys.argv[i].lower() == '-val_perc':
        val_perc = float(sys.argv[i+1])
      if sys.argv[i].lower() == "-epoch_thresh":
        epoch_thresh = float(sys.argv[i+1])
      if sys.argv[i].lower() == "-epoch_max":
        epoch_thresh = int(sys.argv[i+1])
      if sys.argv[i].lower() == "-burnin":
        burnin = int(sys.argv[i+1])
      if sys.argv[i].lower() == "-lrate":
        lrate = float(sys.argv[i+1])
      if sys.argv[i].lower() == "-l2":
        l2 = float(sys.argv[i+1])
      if sys.argv[i].lower() == "-dropout":
        dropout = float(sys.argv[i+1])
      if sys.argv[i].lower() == "-shape":   
        temp_shape = sys.argv[i+1]
        shape_r,shape_c = temp_shape.strip().split(',')
        shape_r = int(shape_r)
        shape_c = int(shape_c)
      if sys.argv[i].lower() == "-conv_shape": 
        temp_shape = sys.argv[i+1]
        conv_r,conv_c = temp_shape.strip().split(',')
        conv_r = int(conv_r)
        conv_c = int(conv_c)
      if sys.argv[i].lower() == "-s_losses":
        save_losses = sys.argv[i+1]
      if sys.argv[i].lower() == "-s_yhat":
        save_yhat = sys.argv[i+1]
      if sys.argv[i].lower() == "-gs_reps":
        gs_reps = int(sys.argv[i+1])



    ################
    ### Features: read in file, keep only those in FEAT if given, and define feature_cols for DNNReg.
    ################
    if os.path.isfile(X_file):
      x = pd.read_csv(X_file, sep=SEP, index_col = 0)
      if FEAT != '':
          with open(FEAT) as f:
              features = f.read().strip().splitlines()
          x = x.loc[:,features]
      
    elif os.path.isdir(X_file):
      x = ANN.fun.Image2Features(X_file, shape_r, shape_c)

    feat_list = list(x.columns)
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
        mean = y.mean(axis=0)
        std = y.std(axis=0)
        y = (y - mean) / std
    y = y.convert_objects(convert_numeric=True)

    df = pd.merge(y, x, left_index=True, right_index=True)

    print('\nSnapshot of data order being used:')
    print(df.head())

    ################
    ### Holdout: Drop holdout set as it will not be used during grid search
    ################
    X, Y, X_train, X_valid, X_test, Y_train, Y_valid, Y_test = ANN.fun.train_valid_test_split(df, ho, y_name, val_perc)


    # TF Graph Placeholders 
    x = tf.placeholder(tf.float32, [None, X_train.shape[1]])
    y = tf.placeholder(tf.float32, [None, 1])
    dropout_rate = tf.placeholder(tf.float32) # For dropout, allows it to be turned on during training and off during testing
    
    if FUNCTION == 'gs' or FUNCTION == 'full':
        print('Starting Grid Search...')
        gs_results = pd.DataFrame()
        gs_count = 0
        gs_length = len(list_dropout) * len(list_l2) * len(list_lrate) * len(list_actfun) * gs_reps
        
        for r in range(0,gs_reps):
          for dropout in list_dropout:
              for l2 in list_l2:
                  for lrate in list_lrate:
                      for actfun in list_actfun:
                            if gs_count % 10 == 0:
                                print('Grid Search Status: %i out of %i' % (gs_count, gs_length))

                            ### Define CNN Model ###
                            pred = ANN.fun.convolutional_neural_network(x, conv_r, conv_c, shape_r, shape_c, dropout, actfun)
                            train_vars = tf.trainable_variables()
                            loss = tf.reduce_mean(tf.squared_difference(pred, Y_train)) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * l2
                            optimizer = tf.train.AdamOptimizer(lrate).minimize(loss)

                            ### Launch the graph ###
                            sess = tf.Session()
                            init = tf.global_variables_initializer()
                            sess.run(init)

                            epoch_count = stop_count = 0
                            train='yes'
                            old_c = 1

                            while train == 'yes':
                              epoch_count += 1
                              _, c = sess.run([optimizer, loss], feed_dict={x:X_train, y:pd.DataFrame(Y_train), dropout_rate:dropout}) # Maybe add keep_prob:dropout to the feed_dict
                              valid_c = sess.run(loss,feed_dict = {x:X_valid, y:pd.DataFrame(Y_valid), dropout_rate:1})

                              pchange = (old_c-valid_c)/old_c
                              if epoch_count >= burnin:
                                  if abs(pchange) < epoch_thresh:
                                      stop_count += 1
                                      print('Early stopping after %i more below threshold' % (10-stop_count))
                                      if stop_count >= 10:
                                          train='no'

                              old_c = valid_c
                              if epoch_count == max_epochs:
                                train='no'

                            # Apply trained network to validation data and gather performance metrics
                            valid_pred = sess.run(pred, feed_dict={x: X_valid, y:pd.DataFrame(Y_valid), dropout_rate:0})
                            val_cor = pearsonr(valid_pred[:,0],Y_valid)
                            gs_results = gs_results.append({'ActFun': actfun, 'dropout': dropout, 'L2':l2, 'lrate':lrate, 'Epochs':epoch_count, 'Train_MSE':c, 'Valid_MSE':valid_c, 'Valid_PCC': val_cor[0]}, ignore_index=True)

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
            
            gs_ave = gs_res.groupby(['ActFun','dropout','L2','lrate']).agg({
                'Valid_Loss': 'median', 'Train_Loss': 'median', 'Valid_PCC': 'mean', 'Epochs': 'mean'}).reset_index()
            gs_ave.columns = ['ActFun','dropout','L2','LRate', 'VLoss_med', 'TLoss_med', 'VPCC_med', 'Epochs_mean']
            results_sorted = gs_ave.sort_values(by='VPCC_med', ascending=False)
            print('\nSnapshot of grid search results:')
            print(results_sorted.head())

            actfun = results_sorted['ActFun'].iloc[0]
            dropout = float(results_sorted['dropout'].iloc[0])
            l2 = float(results_sorted['L2'].iloc[0])
            lrate = float(results_sorted['LRate'].iloc[0])


        print("\n\n##########\nBuilding MLP with the following parameters:\n")
        print('Regularization: dropout = %f  L2 = %f' % (dropout, l2))
        print('Learning rate: %f' % lrate)
        print('Activation Function: %s\n\n\n' % actfun)


        ### Define CNN Model ###
        pred = ANN.fun.convolutional_neural_network(x, conv_r, conv_c, shape_r, shape_c, dropout, actfun)
        train_vars = tf.trainable_variables()
        loss = tf.reduce_mean(tf.squared_difference(pred, Y_train)) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * l2
        optimizer = tf.train.AdamOptimizer(lrate).minimize(loss)

        ### Launch the graph ###
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        epoch_count = stop_count = 0
        train='yes'
        old_c = 1

        while train == 'yes':
          epoch_count += 1
          _, c = sess.run([optimizer, loss], feed_dict={x:X_train, y:pd.DataFrame(Y_train), dropout_rate:dropout}) # Maybe add keep_prob:dropout to the feed_dict
          valid_c = sess.run(loss,feed_dict = {x:X_valid, y:pd.DataFrame(Y_valid), dropout_rate:1})
          test_c = sess.run(loss,feed_dict = {x:X_test, y:pd.DataFrame(Y_test), dropout_rate:1})

          losses.append([epoch_count, c, valid_c, test_c])
          pchange = (old_c-valid_c)/old_c
          if epoch_count >= burnin:
              if abs(pchange) < epoch_thresh:
                  stop_count += 1
                  print('Early stopping after %i more below threshold' % (10-stop_count))
                  if stop_count >= 10:
                      train='no'

          old_c = valid_c
          if epoch_count == max_epochs or train=='no':
            train='no'
            print('Final MSE after %i epochs for training: %.5f and validation: %.5f' % (epoch_count, c, valid_c))

        # Predict test set and add to yhat output
        test_pred = sess.run(pred, feed_dict={x: X_test, dropout_rate:1})
        valid_pred = sess.run(pred, feed_dict={x: X_valid, dropout_rate:1})
        print('Snapshot of predicted Y values:')
        print(test_pred[:,0][0:10])
        ho_cor = np.corrcoef(Y_test, test_pred[:,0])
        valid_cor = np.corrcoef(Y_valid, valid_pred[:,0])
        print('Valid correlation coef (r): %.5f' % valid_cor[0,1])
        print('Holdout correlation coef (r): %.5f' % ho_cor[0,1])



        ##### Optional Outputs ####
        if save_losses == 't':
            losses_df = pd.DataFrame(losses, columns=['epoch', 'MSE_train', 'MSE_valid', 'MSE_test'])        
            losses_df.to_csv(SAVE+'_losses.csv', index=False)

        if save_yhat == 't':
            pred_all = sess.run(pred, feed_dict={x:X, dropout_rate:1})
            pred_all_res = pd.DataFrame({'Y': Y, 'Yhat': pred_all[:,0]})
            pred_all_res.to_csv(SAVE+'_yhat.csv', index=False)
        
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


