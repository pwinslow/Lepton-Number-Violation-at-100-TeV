#########################################################################################################
# This script performs a machine learning pipeline on a given data file. The pipeline trains a 		#
# random forest classifier which is then used to optimize signal and background separation, maximizing  #
# statistical significance of any signal.								#
#########################################################################################################

# Basic imports
from __future__ import division
import sys

# Data analysis imports
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score

# Import the wrangler class to analyze signal files
import Wrangler
reload(Wrangler)
from Wrangler import Wrangler

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')


class MLpipeline(object):
	
	def __init__(self, signal_PATH, data_PATH):
		self.signal_PATH = signal_PATH
		self.data_PATH = data_PATH
		
		
	def read_data(self):
		
		# Read in BG info
		BG_df = pd.read_csv(self.data_PATH + '/BGonly_df.csv')
		cs_df = pd.read_csv(self.data_PATH + '/BGonly_cross_section_and_weights_df.csv', index_col=0)
		
		# Read parameter info
		with open(self.signal_PATH, 'r+') as f:
			param_info = [next(f) for x in range(3)]
		m = float(param_info[1].split(':')[1].strip())
		y = float(param_info[2].split(':')[1].strip())
		
		# Initiate a Wrangler class instance for the signal 
		Signal_Wrangler = Wrangler(Ne_1=2, Np_1=0, Ne_2=0, Np_2=2, Nj=2, Nb=0)
		
		# Define feature list
		Feature_List = ['electron number', 'positron number', 'jet Number', 'H_T(e)', 'H_T(j)', 'Delta_R(L_e, NL_e)', 
		                'Delta_R(L_j, NL_j)', 'Delta_R(L_e, L_j)', 'm(L_e, NL_e)', 'm(L_j, NL_j)', 'm(L_e, L_j)', 
		                'cos(L_e, NL_e)', 'cos(L_j, NL_j)', 'cos(L_e, L_j)', 'MET', 'Sphericity_leptonic', 
		                'Transverse_Sphericity_leptonic', 'Aplanarity_leptonic', 'Planarity_leptonic', 'Sphericity_hadronic', 
		                'Transverse_Sphericity_hadronic', 'Aplanarity_hadronic', 'Planarity_hadronic', 'Sphericity_global', 
		                'Transverse_Sphericity_global', 'Aplanarity_global', 'Planarity_global']
		
		# Import Signal into a dataframe
		X, sigma_Signal, w_Signal, Signal_fraction = Signal_Wrangler.Feature_Architect( self.signal_PATH )
		X_Signal = DataFrame(np.asarray(X))
		X_Signal.columns = Feature_List
		X_Signal['Class'] = 'Signal'
		X_Signal['Subclass'] = 'Signal'
		
		# Reduce BG dataframe to preserve proportion of signal and background events
		all_classes = BG_df['Subclass']
		BGnum = int( round( 5.25 * X_Signal.shape[0] ) )
		redBG_df, BG_drop, Y_keep, Y_drop = train_test_split(BG_df, 
								     all_classes, 
								     train_size = BGnum, 
							 	     random_state = 23, 
								     stratify = all_classes)

		# Adjust cross section weights to account for BG adjustment
		fac = BG_df.shape[0] / redBG_df.shape[0]
		cs_df.loc['cross section weight (pb)'] = cs_df.loc['cross section weight (pb)'] * fac

		# Combine the reduced BG and signal events into one dataframe
		Full_df = pd.concat([redBG_df, X_Signal], axis=0)
		
		# Get list of rows with null values and drop them
		null_list = pd.isnull( Full_df ).any(1).nonzero()[0]
		Full_df.drop( Full_df.index[ null_list ], inplace=True )
		Full_df.index = range(Full_df.shape[0])
		
		# Add signal cross section info to cross section dataframe
		cs_df['Signal'] = Series([sigma_Signal, w_Signal], index=cs_df.index)
				
		return Full_df, cs_df, m, y
		
		
	def feature_normalization(self, df):
		
		# Normalize all non-categorical features
		dont_norm_list = ['jet Number', 'electron number', 'positron number']
		features_to_be_normed = [x for x in df.columns if x not in dont_norm_list][:-2]
		df[ features_to_be_normed ] = df[ features_to_be_normed ].apply(lambda x: x / x.max(), axis=0)
		
		# Generate dummy variables for jets
		jet_dummies = pd.get_dummies(df['jet Number'], prefix='Jet_Number:')
		df = pd.concat([jet_dummies, df], axis=1)
		df.drop('jet Number', axis=1, inplace=True)

		# Generate dummy variables for electrons
		electron_dummies = pd.get_dummies(df['electron number'], prefix='Electron_Number:')
		df = pd.concat([electron_dummies, df], axis=1)
		df.drop('electron number', axis=1, inplace=True)
		
		# Generate dummy variables for positrons
		positron_dummies = pd.get_dummies(df['positron number'], prefix='Positron_Number:')
		df = pd.concat([positron_dummies, df], axis=1)
		df.drop('positron number', axis=1, inplace=True)
		
		return df


	# Define a short method to reclassify signal (all backgrounds) as the '1' ('0') class
	def reclass(self, Y):
	    Y_new = np.copy(Y)
	    Y_new[Y_new != 'Signal'] = 0
	    Y_new[Y_new == 'Signal'] = 1
	    Y_new = Y_new.astype(int)
	    return Y_new
					
					
	def feature_selection(self, X_dev, Y_dev, X_eval):
		
		# Perform feature selection using a tree-based estimator (threshold for selection the importance mean)
		clf = ExtraTreesClassifier(n_estimators=200)
		clf.fit(X_dev, Y_dev)
		model = SelectFromModel(clf, prefit=True)
		
		Xdev = model.transform(X_dev)
		Xeval = model.transform(X_eval)
		
		return model, Xdev, Xeval
		
		
	# Make sure to perform feature selection on X before using this method
	def grid_optimization(self, X, Y):
		
		# Initialize a random forest classifier
		rf = RandomForestClassifier()
		
		# Define a parameter grid to search over
		param_dist = {"n_estimators": range(50, 550, 50), 
		              "max_depth": range(3, 17, 2), 
		              "criterion": ['gini', 'entropy']}	
																
		# Setup 10-fold stratified cross validation
		cross_validation = StratifiedKFold(Y, n_folds=10)
		
		# Reclassify class vector
		Y = self.reclass(Y)
		
		# Randomly sample 10 hyperparameter configurations from the grid above and perform 5-fold cross validation for each
		n_iter_search = 15
		clf = RandomizedSearchCV(rf, 
		                  	 param_distributions = param_dist, 
		                  	 cv = cross_validation, 
				  	 n_iter = n_iter_search,
		                  	 n_jobs = 10,
				  	 verbose = 10)

		# Fit the model and return the optimized hyperparameter configuration
		clf.fit(X, Y)		
		rf_best = clf.best_estimator_
		
		return rf_best
		
	
	# Inputs should already reflect feature selection
	# cs_df weights should be rescaled so cross sections span X_eval events
	def calculate_significance(self, Xeval, Yeval, cs_df, rf_model):
		
		# Collect histogram counts for the two-class decision scores
		twoclass_output = rf_model.predict_proba(Xeval)[:, 1]
		cnts_list = []
		for i in range(2):
		    cnts, bins, bars = plt.hist(twoclass_output[ self.reclass(Yeval) == i ], bins=50)
		    plt.clf()
		    cnts_list.append(cnts)

		# Calculate the optimized cut on the decision score which leads to a signal-dominated spectrum
		cut = bins[ cnts_list[0] < cnts_list[1] ][0]
		
		# Create new dataframe, characterising cross section weights and decision scores
		weight_map = {}
		for source in cs_df.columns:
		    weight_map[source] = cs_df[source]['cross section weight (pb)']
		df = pd.concat([Series(Yeval), Series(twoclass_output, index=Yeval.index)], axis=1)
		df.columns = ['Subclass', 'Decision_Score']
		df['Weight'] = df.Subclass.map(weight_map)
		df['Class'] = df.Subclass.map(lambda x: 'Background' if x != 'Signal' else 'Signal')
		df = df[['Class', 'Subclass', 'Weight', 'Decision_Score']]

		# Select only events from the signal dominated region and calculate the corresponding poisson significance.
		# Luminosity is assumed to be in inverse femtobarns.
		cut_df = df[ df.Decision_Score >= cut ]
		B = cut_df[ cut_df.Class == 'Background' ].Weight.sum()
		S = cut_df[ cut_df.Class == 'Signal' ].Weight.sum()
		Z = np.sqrt(1e+3) * np.sqrt(2 * (S+B) * np.log(1 + S/B) - 2 * S)

		# Return optimized poisson significance
		return Z


	# All datasets input here should have gone through feature selection
	def classification_report(self, run, Xeval, Yeval, rf_model):
		
		Y_predicted = rf_model.predict(Xeval)
		Y_eval = self.reclass(Yeval)
		roc_auc = roc_auc_score(Y_eval, Y_predicted)
		
		report = classification_report(Y_eval, 
		                               Y_predicted,
		                               target_names=["background", "signal"])
		roc = "Area under ROC curve: {:0.3f}".format(roc_auc)

		return 'Classification report for run{0}:\n{1}{2}\n\n'.format(run, report, roc)


if __name__ == '__main__':
	
	# Read in path to signal info and initialize an instance of the MLpipeline class
	signal_PATH = sys.argv[1]
	data_PATH = sys.argv[2]
	runNum = sys.argv[3]
	pipeline = MLpipeline(signal_PATH, data_PATH)
	
	# Read in data
	Full_df, cs_df, m, y = pipeline.read_data()
	
	# Normalize features
	Full_df = pipeline.feature_normalization( Full_df )
	
	# Split the dataset into development and evaluation sets
	all_inputs = Full_df[Full_df.columns[:-2]]
	all_classes = Full_df['Subclass']
	X_dev, X_eval, Y_dev, Y_eval = train_test_split(all_inputs, 
	                                                all_classes, 
	                                                test_size = 0.33, 
	                                                random_state = 42, 
	                                                stratify = all_classes)									

	# Perform feature selection for both development and evaluation sets
	fs_model, selected_Xdev, selected_Xeval = pipeline.feature_selection(X_dev, Y_dev, X_eval)

	# Derive optimized random forest model
	rf_model = pipeline.grid_optimization(selected_Xdev, Y_dev)
	
	# Adjust cross section weights so that cross sections span only the evaluation set
	fac = Full_df.shape[0] / X_eval.shape[0]
        cs_df.loc['cross section weight (pb)'] = cs_df.loc['cross section weight (pb)'] * fac

	# Use random forest to calculate the cutoff in decision score spectrum where signal begins to dominate background
	Z = pipeline.calculate_significance(selected_Xeval, Y_eval, cs_df, rf_model)

	# Record signal cross section and weight
	sigma, weight = cs_df['Signal']['cross section (pb)'], cs_df['Signal']['cross section weight (pb)']
	
	# Record classification reports
	with open('/data2/pwinslow/100TeV_Scan_Results/extra_Reports.dat', 'a+') as report:
		report.write( pipeline.classification_report(runNum, selected_Xeval, Y_eval, rf_model) )	

	# Record results
	with open('/data2/pwinslow/100TeV_Scan_Results/extra_Results.csv', 'a+') as output:
		output.write('{0}, {1}, {2}, {3}, {4}, {5}\n'.format(runNum, m, y, sigma, weight, Z))
