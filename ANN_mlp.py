from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, argparse, timeit
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import ANN_Functions as ANN

start_time = timeit.default_timer()
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def main():
    print('Running ANN_MLP pipeline...')

    ########################
    ### Parse Input Args ###
    ########################

    parser = argparse.ArgumentParser(description='Build Multilayer Perceptron Artificial Neural Networks implemented in TensorFlow (TF)')
    
    # Input arguments
    parser.add_argument('-FUNCTION', '-f', help='Select function to perform (gs, run, full)', default='full')
    parser.add_argument('-X_file', '-x', help='Features file', required=True)
    parser.add_argument('-Y_file', '-y', help='Value to predict file', required=True)
    parser.add_argument('-SEP', '-sep', help='Deliminator', default='\t')
    parser.add_argument('-y_name', help='Name of column in Y_file to predict', default='Y')
    parser.add_argument('-norm', help='T/F to normalize Y values', default='t')
    parser.add_argument('-ho', help='File with testing (i.e. holdout) lines', required=True)
    parser.add_argument('-FEAT', '-feat', help='File with list of features (from x) to include', default='')

    # Output arguments
    parser.add_argument('-save', help='prefix for output files', default='output')
    parser.add_argument('-TAG', '-tag', help='Identifier string to add to RESULTS output line', default='')
    parser.add_argument('-out_loc', help='Path to where output files are saved. Default to cwd.', default='')
    parser.add_argument('-save_weights', '-s_weights', help='T/F Save the trained weights from the trained network (only if hidden layers <= 3)', default='f')
    parser.add_argument('-save_losses', '-s_losses', help='T/F Save the training, validation, and testing losses from final model training', default='f')
    parser.add_argument('-save_yhat', '-s_yhat', help='T/F Apply trained model to all data and save output', default='f')

    # Default Hyperparameters
    parser.add_argument('-params', help='Output from -f gs (i.e. SAVE_GridSearch.txt)', default='')
    parser.add_argument('-actfun', help='Activation function. (relu, sigmoid)', default='sigmoid')
    parser.add_argument('-lrate', help='Learning Rate', default=0.01, type=float)
    parser.add_argument('-dropout', help='Dropout rate', default=0.1, type=float)
    parser.add_argument('-l2', help='Shrinkage parameter for L2 regularization', default=0.0, type=float)
    parser.add_argument('-arc', help='MLP architecture as comma separated layer sizes (e.g. 100,50 or 200,100,50)', default='10,5')

    # Grid Search reps/space
    parser.add_argument('-gs_reps', '-gs_n', help='Number of Grid Search Reps (will append results if SAVE_GridSearch.csv exists)', type=int, default=10)
    parser.add_argument('-actfun_gs', help='Activation functions for Grid Search', nargs='*', type=str, default=["sigmoid", "relu"])
    parser.add_argument('-dropout_gs', help='Dropout rates for Grid Search', nargs='*', type=float, default=[0.1, 0.5])
    parser.add_argument('-l2_gs', help='Shrinkage parameters for L2 for Grid Search', nargs='*', type=float, default=[0.0, 0.1, 0.5])
    parser.add_argument('-lrate_gs', help='Shrinkage parameters for L2 for Grid Search', nargs='*', type=float, default=[0.01, 0.001])
    parser.add_argument('-arc_gs', help='Architectures for Grid Search', nargs='*', type=str, default=["10", "50", "100", "10,5", "50,25", "100,50", "10,5,5", "50,25,10"])
   
    # Training behavior 
    parser.add_argument('-n_reps', '-n', help='Number of replicates (unique validation set/starting weights for each)', default=100, type=int)
    parser.add_argument('-burnin', help='Number of epochs before start counting for early stopping', default=100, type=int)
    parser.add_argument('-epoch_thresh', help='Threshold for percent change in MSE for early stopping', default='0.001', type=float)
    parser.add_argument('-max_epochs', help='Max number of epochs to iterate through', default=50000, type=int)
    parser.add_argument('-val_perc', help='Percent of the training set to use for validation', default=0.1, type=float)
    parser.add_argument('-loss_type', help='Loss function to minimize during training. Only MSE available now', default='mse', type=str)
    parser.add_argument('-WEIGHTS', '-weights', help='Approach for starting weights (random, xavier, RF, rrB, BayesA, BayesB, BL, BRR)', default='xavier', type=str)
    parser.add_argument('-weight_mu', '-mu', help='Mean of noise to add to starting weights (not for random/xavier)', default=0.0, type=float)
    parser.add_argument('-weight_sigma', '-sigma', help='Stdev of noise to add to starting weights (not for random/xavier)', default=0.01, type=float)

    run_again = rerun_na = 't'
    rerun_count = 0

    args = parser.parse_args()

    if args.out_loc != '':
        args.save = args.out_loc + '/' + args.save




    ################
    ### Features: read in file, keep only those in FEAT if given, and define feature_cols for DNNReg.
    ################
    x = pd.read_csv(args.X_file, sep=args.SEP, index_col = 0)
    if args.FEAT != '':
        with open(args.FEAT) as f:
            features = f.read().strip().splitlines()
        x = x.loc[:,features]
    feat_list = list(x.columns)
    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in feat_list]

    print("\n\nTotal number of instances: %s" % (str(x.shape[0])))
    print("\nNumber of features used: %s" % (str(x.shape[1])))

    ################
    ### Y: read in file, keep only column to predict, normalize if needed, and merge with features
    ################
    y = pd.read_csv(args.Y_file, sep=args.SEP, index_col = 0)
    if args.y_name != 'pass':
        print('Building model to predict: %s' % str(args.y_name))
        y = y[[args.y_name]]
    if args.norm == 't':
        print('Normalizing Y...')
        mean = y.mean(axis=0)
        std = y.std(axis=0)
        y = (y - mean) / std

    df = pd.merge(y, x, left_index=True, right_index=True)
    yhat = df[args.y_name]

    print('\nSnapshot of data being used:')
    print(df.head())

    ################
    ### Train/Validation/Test Split
    ################
    print('Removing holdout instances to apply model on later...')

    X, Y, X_train, X_valid, X_test, Y_train, Y_valid, Y_test = ANN.fun.train_valid_test_split(df, args.ho, args.y_name, args.val_perc)
    
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

    if args.FUNCTION in ['gs', 'full', 'both']:
        print('Starting Grid Search...')
        gs_results = pd.DataFrame()
        gs_count = 0
        gs_length = len(args.dropout_gs) * len(args.l2_gs) * len(args.lrate_gs) * len(args.actfun_gs) * len(args.arc_gs) * args.gs_reps
        for r in range(0,args.gs_reps):
            X, Y, X_train, X_valid, X_test, Y_train, Y_valid, Y_test = ANN.fun.train_valid_test_split(df, args.ho, args.y_name, args.val_perc)
            for arc in args.arc_gs:
                archit, layer_number = ANN.fun.define_architecture(arc)
                weights, biases = ANN.fun.initialize_starting_weights(args.WEIGHTS, n_input, n_classes, archit, layer_number, df, args.weight_mu, args.weight_sigma, X_train, Y_train)
                for l2 in args.l2_gs:
                    for lrate in args.lrate_gs:
                        for actfun in args.actfun_gs:
                            for dropout in args.dropout_gs:
                                if gs_count % 10 == 0:
                                    print('Grid Search Status: %i out of %i' % (gs_count, gs_length))
                                
                                # Construct ANN model
                                
                                pred = ANN.fun.multilayer_perceptron(x, weights, biases, layer_number, actfun, dropout)
                                loss = ANN.fun.define_loss(args.loss_type, y, pred, l2, weights)
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
                                    
                                    if epoch_count >= args.burnin:
                                        pchange = (old_c-valid_c)/old_c
                                        if abs(pchange) < args.epoch_thresh:
                                            stop_count += 1
                                            if stop_count >= 10:
                                                train='no'
                                    old_c = valid_c
                                    if epoch_count == args.max_epochs:
                                        train='no'

                                valid_pred = sess.run(pred, feed_dict = {x:X_valid, y:pd.DataFrame(Y_valid), dropout_rate:1})
                                val_cor = np.corrcoef(Y_valid, valid_pred[:,0])
                                gs_results = gs_results.append({'ActFun':actfun, 'Arch':arc, 'dropout':dropout, 'L2':l2, 'LearnRate':lrate,'Epochs':epoch_count, 'Train_Loss':c, 'Valid_Loss':valid_c, 'Valid_PCC':val_cor[0,1]}, ignore_index=True)
                                gs_count += 1
            
        if not os.path.isfile(args.save + "_GridSearch.txt"):
            gs_results.to_csv(args.save + "_GridSearch.txt", header='column_names', sep='\t')
        else: 
            gs_results.to_csv(args.save + "_GridSearch.txt", mode='a', header=False, sep='\t')

        print('\n\n Grid Search results saved to: %s_GridSearch.txt\n' % args.save)


    
    ################
    ### Run final model 
    ################

    print('####### Running Final Model(s) ###########')

    if args.FUNCTION in ['full','both','run']:


        
        # Grab parameters from grid search results
        while run_again == 't':
            
            if args.FUNCTION in ['full', 'both'] or args.params != '': # Else use default or individually defined parameters
                run_again = 'f'

                if args.FUNCTION in ['full', 'both']:
                    gs_res = gs_results
                
                elif args.params != '':
                    gs_res = pd.read_csv(args.params, sep='\t')


                gs_res.fillna(0, inplace=True)
                gs_ave = gs_res.groupby(['ActFun','dropout','L2','Arch','LearnRate']).agg({
                    'Valid_Loss': 'median', 'Train_Loss': 'median', 'Valid_PCC': 'median', 'Epochs': 'mean'}).reset_index()
                gs_ave.columns = ['ActFun','dropout','L2','Arch','LRate', 'VLoss_med', 'TLoss_med', 'VPCC_med', 'Epochs_mean']
                results_sorted = gs_ave.sort_values(by='VPCC_med', ascending=False)
                print('\nSnapshot of grid search results:')
                results_sorted = results_sorted[(results_sorted.dropout > 0) | (results_sorted.L2 > 0)]
                print(results_sorted.head())

                actfun = results_sorted['ActFun'].iloc[0]
                dropout = float(results_sorted['dropout'].iloc[0])
                l2 = float(results_sorted['L2'].iloc[0])
                lrate = float(results_sorted['LRate'].iloc[0])
                arc = results_sorted['Arch'].iloc[0]

            elif args.FUNCTION == 'run' and args.params == '':
                actfun = args.actfun
                dropout = args.dropout
                l2 = args.l2
                lrate = args.lrate
                arc = args.arc


            print("\n\n##########\nBuilding MLP with the following parameters:\n")
            print('Architecture: %s' % arc)
            print('Regularization: dropout = %f  L2 = %f' % (dropout, l2))
            print('Learning rate: %f' % lrate)
            print('Activation Function: %s\n\n\n' % actfun)

            yhats_reps = df[args.y_name]
            for n in range(args.n_reps):
                print("\nModel replicate: %i" % n)
                X, Y, X_train, X_valid, X_test, Y_train, Y_valid, Y_test = ANN.fun.train_valid_test_split(df, args.ho, args.y_name, args.val_perc)

                # Construct ANN model
                archit, layer_number = ANN.fun.define_architecture(arc)
                weights, biases = ANN.fun.initialize_starting_weights(args.WEIGHTS, n_input, n_classes, archit, layer_number, df, args.weight_mu, args.weight_sigma, X_train, Y_train)

                pred = ANN.fun.multilayer_perceptron(x, weights, biases, layer_number, actfun, dropout)
                loss = ANN.fun.define_loss(args.loss_type, y, pred, l2, weights)
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
                    if epoch_count >= args.burnin:
                        if abs(pchange) < args.epoch_thresh:
                            stop_count += 1
                            print('Early stopping after %i more below threshold' % (10-stop_count))
                            if stop_count >= 10:
                                train='no'

                    if (epoch_count) % 10 == 0:
                        print("Epoch:", '%i' % (epoch_count), "; Training MSE=", "{:.3f}".format(c), "; Valid MSE=", "{:.3f}".format(valid_c), '; Percent change=', str(pchange))
         
                    old_c = valid_c
                    if epoch_count == args.max_epochs or train=='no':
                        print('Final MSE after %i epochs for training: %.5f, validation: %.5f, and testing: %.5f' % (epoch_count, c, valid_c, test_c))
                        train = 'no'

                # Predict test set and add to yhat output
                test_pred = sess.run(pred, feed_dict={x: X_test, dropout_rate:1})
                valid_pred = sess.run(pred, feed_dict={x: X_valid, dropout_rate:1})
                print('Snapshot of predicted Y values:')
                print(test_pred[:,0][0:10])
                ho_cor = np.corrcoef(Y_test, test_pred[:,0])
                valid_cor = np.corrcoef(Y_valid, valid_pred[:,0])
                print('Valid correlation coef (r): %.5f' % valid_cor[0,1])
                print('Holdout correlation coef (r): %.5f' % ho_cor[0,1])

                if rerun_na == 't':
                    if np.isnan(ho_cor[0,1]):
                        rerun_count += 1
                        if rerun_count <= 10:
                            print('\n\n!!! Model predicted same y for all instances... repeating initalization & training...\n')
                            run_again = 't'
                        else:
                            print('Did not converge on solution after 10 trys... Exiting...')
                            quit()
                    if valid_cor[0,1] <= 0:
                        rerun_count += 1
                        if rerun_count <= 10:
                            print('\n\n!!! Model had negative correlation with validation set... repeat...\n')
                            run_again = 't'
                    else:
                        run_again = 'f'




                ##### Optional Outputs ####

                if args.save_weights == 't':
                    ANN.fun.save_trained_weights(args.save, tvars, tvars_vals, archit, feat_list)
                    if args.n_reps > 1:
                        print("SAVING WEIGHTS - NOTE: DOES NOT AVERAGE OVER N_REPS - WILL OVERRIGHT! FIX LATER!?!?")

                if args.save_losses == 't':
                    losses_df = pd.DataFrame(losses, columns=['epoch', 'MSE_train', 'MSE_valid', 'MSE_test'])        
                    losses_df.to_csv(args.save+'_losses.csv', index=False)
                    if args.n_reps > 1:
                        print("SAVING LOSSES - NOTE: DOES NOT AVERAGE OVER N_REPS - WILL OVERRIGHT AND REPORT LAST REP")

                if args.save_yhat == 't':
                    tmp_yhat = sess.run(pred, feed_dict={x:X, dropout_rate:1})
                    rep_name = 'yhat_' + str(int(n) + 1)
                    tmp_yhat_df = pd.DataFrame(data=tmp_yhat[:,0], index=df.index, columns = [rep_name])
                    yhats_reps = pd.concat([yhats_reps,tmp_yhat_df], axis = 1)
                
                run_time = timeit.default_timer() - start_time
                if not os.path.isfile('RESULTS.txt'):
                    out1 = open('RESULTS.txt', 'w')
                    out1.write('DateTime\tRunTime\tTag\tDFs\tDFy\tTrait\tFeatSel\tWeights\tNumFeat\tHoldout\tNumHidLay\tArchit\tActFun\tEpochs\tdropout\tL2\tLearnRate\tMSE_Train\tMSE_Valid\tMSE_test\tPCC_test\n')
                    out1.close()
                     
                out2 = open('RESULTS.txt', 'a')
                out2.write('%s\t%0.5f\t%s\t%s\t%s\t%s\t%s\t%s\t%i\t%s\t%i\t%s\t%s\t%i\t%f\t%f\t%f\t%0.5f\t%0.5f\t%0.5f\t%0.5f\n' % (
                    timestamp, run_time, args.TAG, args.X_file, args.Y_file, args.y_name, args.FEAT, args.WEIGHTS, x.shape[1], args.ho, layer_number, str(arc), actfun, epoch_count, dropout, l2, lrate, c, valid_c, test_c, ho_cor[0,1]))
                out2.close()
        
        if args.save_yhat == 't':
            yhat_cols = [c for c in yhats_reps.columns if c.startswith('yhat')]
            yhats_reps.insert(loc=1, column = 'Mean', value = yhats_reps[yhat_cols].mean(axis=1))
            yhats_reps.insert(loc=2, column = 'Median', value = yhats_reps[yhat_cols].median(axis=1))
            yhats_reps.insert(loc=3, column = 'stdev', value = yhats_reps[yhat_cols].std(axis=1))

            yhats_reps.to_csv(args.save+'_yhat.csv', index=True)
                
        print('\nfinished!')


if __name__ == '__main__':
    main()