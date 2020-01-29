# Import statements
# General:
import pandas as pd
import numpy as np

# rdkit:
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.AllChem import GetHashedTopologicalTorsionFingerprintAsBitVect
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem import PandasTools
from rdkit.Chem.FilterCatalog import *

# sklearn:
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
# from sklearn.manifold import MDS

# matplotlib:
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# seaborn:
import seaborn as sns

# others:
from chembl_webresource_client.new_client import new_client
import os
import math
import argparse
import warnings


def define_args():
	parser = argparse.ArgumentParser(description='Collect affinity data for a protein kinase target')
	parser.add_argument('--uid', type=str, help='Enter Uniprot-ID')
	parser.add_argument('--output', type=str, help='Enter directory for output files')
	parser.add_argument('--pains', type=str2bool, default=True, help='Activate PAINS-filter. Default: True')
	parser.add_argument('--ml', type=str2bitlist, help='Set bitstring for machine learning algorithms for RF, SVM, MLP, kNN and GCP (in this order) like 11010')
	return parser.parse_args()


def str2bool(v):
	if isinstance(v, bool):
	   return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def str2bitlist(v):
	if len(v) != 5: raise argparse.ArgumentTypeError('Exact five boolean values expected.')

	bitlist = []
	for b in v:
		if b == '1':
			bitlist.append(True)
		elif b == '0':
			bitlist.append(False)
		else:
			raise argparse.ArgumentTypeError('Boolean values expected.')
	return bitlist


def convert_to_NM(unit, bioactivity):
	# c=0
	# for i, unit in enumerate(bioact_df.units):
	if unit != "nM":        
		if unit == "pM":
			value = float(bioactivity)/1000
		elif unit == "10'-11M":
			value = float(bioactivity)/100
		elif unit == "10'-10M":
			value = float(bioactivity)/10
		elif unit == "10'-8M":
			value = float(bioactivity)*10
		elif unit == "10'-1microM" or unit == "10'-7M":
			value = float(bioactivity)*100
		elif unit == "uM" or unit == "/uM" or unit == "10'-6M":
			value = float(bioactivity)*1000
		elif unit == "10'1 uM":
			value = float(bioactivity)*10000
		elif unit == "10'2 uM":
			value = float(bioactivity)*100000
		elif unit == "mM" or "microM":
			value = float(bioactivity)*1000000
		elif unit == "M":
			value = float(bioactivity)*1000000000
		else:
			print ('unit not recognized...', unit)
		return value
	else: return bioactivity


def calculate_fp(mol, method='maccs', n_bits=2048):
	# mol = Chem molecule object
	# Function to calculate molecular fingerprints given the number of bits and the method
	if method == 'maccs':
		return MACCSkeys.GenMACCSKeys(mol)
	if method == 'ecfp4':
		return GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits, useFeatures=False)
	if method == 'ecfp6':
		return GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits, useFeatures=False)
	if method == 'torsion':
		return GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
	if method == 'rdk5':
		return RDKFingerprint(mol, maxPath=5, fpSize=1024, nBitsPerHash=2)


def create_mol(df_l, n_bits):
	# Construct a molecule from a SMILES string
	# Generate mol column: Returns a Mol object, None on failure.
	df_l['mol'] = df_l.smiles.apply(Chem.MolFromSmiles)
	# Create a column for storing the molecular fingerprint as fingerprint object
	df_l['bv'] = df_l.mol.apply(
		# Apply the lambda function "calculate_fp" for each molecule
		lambda x: calculate_fp(x, 'maccs', n_bits)
	)
	# Allocate np.array to hold fp bit-vector (np = numpy)
	df_l['np_bv'] = np.zeros((len(df_l), df_l['bv'][0].GetNumBits())).tolist()
	df_l.np_bv = df_l.np_bv.apply(np.array)
	# Convert the object fingerprint to NumpyArray and store in np_bv
	df_l.apply(lambda x: ConvertToNumpyArray(x.bv, x.np_bv), axis=1)


# Function for a cross-validation loop.
def crossvalidation(model_l, df_l, n_folds=10):
	# Given the selected model, the dataFrame and the number of folds the function executes a crossvalidation and returns
	# accuracy, sensitivity, specificity for the prediction as well as fpr, tpr, roc_auc for each fold
	# Empty results vector
	results = []
	# Shuffle the indices for the k-fold cross-validation
	kf = KFold(n_splits=n_folds, shuffle=True)
	# Labels initialized with -1 for each data-point
	labels = -1 * np.ones(len(df_l))
	# Loop over the folds
	for train_index, test_index in kf.split(df_l):
		# Training
		# Convert the bit-vector and the label to a list
		train_x = df_l.iloc[train_index].bv.tolist()
		train_y = df_l.iloc[train_index].active.tolist()
		# Fit the model
		model_l.fit(train_x, train_y)
		# Testing
		# Convert the bit-vector and the label to a list
		test_x = df_l.iloc[test_index].bv.tolist()
		test_y = df_l.iloc[test_index].active.tolist()
		# Predict on test-set
		prediction_prob = model_l.predict_proba(test_x)[:, 1]
		# Save the predicted label of each fold
		labels[test_index] = model_l.predict(test_x)
		# Performance
		# Get fpr, tpr and roc_auc for each fold
		fpr_l, tpr_l, _ = roc_curve(test_y, prediction_prob)
		roc_auc_l = auc(fpr_l, tpr_l)
		# Append to results
		results.append((fpr_l, tpr_l, roc_auc_l))
	# Get overall accuracy, sensitivity, specificity
	y = df_l.active.tolist()
	acc = accuracy_score(df_l.active.tolist(), labels)
	prec = precision_score(df_l.active.tolist(), labels)
	sens = recall_score(df_l.active.tolist(), labels)
	spec = (acc * len(y) - sens * sum(y)) / (len(y) - sum(y))
	return acc, prec, sens, spec, results


def print_results(acc, prec, sens, spec, stat_res, main_text, file_name, output_dir, plot_figure=1):
	plt.figure(plot_figure, figsize=(7, 7))
	cmap = cm.get_cmap('Blues')
	colors = [cmap(i) for i in np.linspace(0.3, 1.0, 10)]
	#colors = ["#3465A4"]
	for i, (fpr_l, tpr_l, roc_auc_l) in enumerate(stat_res):
		plt.plot(fpr_l, tpr_l, label='AUC CV$_{0}$ = {1:0.2f}'.format(str(i),roc_auc_l), lw=2, color=colors[i])
		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
	plt.plot([0, 1], [0, 1], linestyle='--', label='Random', lw=2, color="black")  # Random curve
	plt.xlabel('False positive rate', size=24)
	plt.ylabel('True positive rate', size=24)
	plt.title(main_text, size=24)
	plt.tick_params(labelsize=16)
	plt.legend(fontsize=16)
	# create dir if not exists
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	# Save plot - use bbox_inches to include text boxes:
	# https://stackoverflow.com/questions/44642082/text-or-legend-cut-from-matplotlib-figure-on-savefig?rq=1
	plt.savefig(output_dir + "/" + file_name, dpi=300, bbox_inches="tight", transparent=True)
	plt.close()
	# Calculate mean AUC and print
	m_auc = np.mean([elem[2] for elem in stat_res])
	print('Mean AUC: {}'.format(m_auc))
	# Show overall accuracy, sensitivity, specificity
	print('Accuracy: {}\nPrecision: {}\nSensitivity: {}\nSpecificity: {}\n'.format(acc, prec, sens, spec))


def bioactivity(chembl_id, bioactivities):
	bioact = bioactivities.filter(target_chembl_id = chembl_id) \
					  .filter(type = 'IC50') \
					  .filter(relation = '=') \
					  .filter(assay_type = 'B') \
					  .only('activity_id','assay_chembl_id', 'assay_description', 'assay_type', \
							'molecule_chembl_id', 'type', 'units', 'relation', 'value', \
							'target_chembl_id', 'target_organism')
	bioact_df = pd.DataFrame.from_records(bioact)
	bioact_df = bioact_df.dropna(axis=0, how = 'any')
	bioact_df = bioact_df.drop_duplicates('molecule_chembl_id', keep = 'first')
	bioact_df = bioact_df.drop(bioact_df.index[~bioact_df.units.str.contains('M')])
	bioact_df = bioact_df.reset_index(drop=True)
	bioactivity_nM = []
	for i, row in bioact_df.iterrows():
		bioact_nM = convert_to_NM(row['units'], row['value'])
		bioactivity_nM.append(bioact_nM)
	bioact_df['value'] = bioactivity_nM
	bioact_df['units'] = 'nM'
	return bioact_df


def compound(bioact_df, compounds):
	cmpd_id_list = list(bioact_df['molecule_chembl_id'])
	compound_list = compounds.filter(molecule_chembl_id__in = cmpd_id_list).only('molecule_chembl_id','molecule_structures')
	compound_df = pd.DataFrame.from_records(compound_list)
	compound_df = compound_df.drop_duplicates('molecule_chembl_id', keep = 'first')
	for i, cmpd in compound_df.iterrows():
		if compound_df.loc[i]['molecule_structures'] != None:
			compound_df.loc[i]['molecule_structures'] = cmpd['molecule_structures']['canonical_smiles']
	output_df = pd.merge(bioact_df[['molecule_chembl_id','units','value']], compound_df, on='molecule_chembl_id')
	output_df = output_df.rename(columns= {'molecule_structures':'smiles', 'value':'IC50'})
	output_df = output_df[~output_df['smiles'].isnull()]
	output_df = output_df.reset_index(drop=True)
	ic50 = output_df.IC50.astype(float) 
	# Convert IC50 to pIC50 and add pIC50 column:
	pIC50 = pd.Series() 
	i = 0
	while i < len(output_df.IC50):
		value = 9 - math.log10(ic50[i]) # pIC50=-log10(IC50 mol/l) --> for nM: -log10(IC50*10**-9)= 9-log10(IC50)
		if value < 0:
			print("Negative pIC50 value at index"+str(i))
		pIC50.at[i] = value
		i += 1
	output_df['pIC50'] = pIC50
	return output_df


def df_rule_of_five(df):
	smi = df['smiles']
	m = Chem.MolFromSmiles(smi)
	# Calculate rule of five chemical properties
	MW = Descriptors.ExactMolWt(m)
	HBA = Descriptors.NumHAcceptors(m)
	HBD = Descriptors.NumHDonors(m)
	LogP = Descriptors.MolLogP(m)
	# Rule of five conditions
	conditions = [MW <= 500, HBA <= 10, HBD <= 5, LogP <= 5]
	# Create pandas row for conditions results with values and information whether rule of five is violated
	return pd.Series([MW, HBA, HBD, LogP, 'yes']) if conditions.count(True) >= 3 else pd.Series([MW, HBA, HBD, LogP, 'no'])


def filter_rule_of_five(output_df):
	# Apply ruleOfFive to dataset to get rule of five results (may take a while)
	rule5_prop_df = output_df.apply(df_rule_of_five, axis=1)
	# Name condition columns
	rule5_prop_df.columns= ['MW', 'HBA', 'HBD', 'LogP', 'rule_of_five_conform']
	# Concatenate dataset with computed values
	output_df = output_df.join(rule5_prop_df)
	# Delete empty rows --> rule of five
	filtered_df = output_df[output_df['rule_of_five_conform']=='yes']
	return filtered_df


def pains(filtered_df):
	filteredData = filtered_df
	params = FilterCatalogParams()
	# Build a catalog from all PAINS (A, B and C)
	params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
	catalog = FilterCatalog(params)
	# Create empty dataframes for filtered data
	rdkit_highLightFramePAINS = pd.DataFrame(columns=('CompID', 'CompMol', 'unwantedID'))
	rdkit_noPAINS = pd.DataFrame(columns=('ChEMBL_ID', 'smiles','pIC50'))
	rdkit_withPAINS = pd.DataFrame(columns=('ChEMBL_ID', 'smiles', 'pIC50','unwantedID'))
	# For index and row in the filtered df
	for i,row in filteredData.iterrows():
		curMol = Chem.MolFromSmiles(row.smiles) # Current molecule
		match = False # Set match to false
		rdkit_PAINSList = []
		# Get the first match
		entry = catalog.GetFirstMatch(curMol)
		if entry!=None:
			# Add name of current unwanted subsftructure to list
			rdkit_PAINSList.append(entry.GetDescription().capitalize())
			# Add relevant matching information to dataframe
			rdkit_highLightFramePAINS.loc[len(rdkit_highLightFramePAINS)] = [row.molecule_chembl_id, curMol,
			entry.GetDescription().capitalize()]
			match = True
		if not match:
			# Add to frame of PAINS free compounds
			rdkit_noPAINS.loc[len(rdkit_noPAINS)] = [row.molecule_chembl_id, row.smiles, row.pIC50]
		else: 
			# Add to frame of compounds that contain PAINS
			# Put the relevant information in the dataframe with the unwanted substructures
			rdkit_withPAINS.loc[len(rdkit_withPAINS)] = [row.molecule_chembl_id, row.smiles, row.pIC50, entry.GetDescription().capitalize()]
	df = rdkit_noPAINS
	# Drop unnecessary columns
	## df_new = df.drop(['units', 'IC50'], axis=1)
	df_new = df
	# Create molecules from smiles and their fingerprints
	create_mol(df_new, 2048)
	# Add column for activity
	df_new['active'] = np.zeros(len(df_new))
	# Mark every molecule as active with an pIC50 of > 6.3
	df_new.loc[df_new[df_new.pIC50 >= 6.3].index, 'active'] = 1.0
	return df_new


def no_pains(filtered_df):
	df_new = filtered_df.drop(['units', 'IC50', 'MW', 'HBA', 'HBD', 'LogP', 'rule_of_five_conform'], axis=1)
	# Create molecules from smiles and their fingerprints
	create_mol(df_new, 2048)
	# Add column for activity
	df_new['active'] = np.zeros(len(df_new))
	# Mark every molecule as active with an pIC50 of > 6.3
	df_new.loc[df_new[df_new.pIC50 >= 6.3].index, 'active'] = 1.0
	return df_new


def ml_algorithms(df_new, args):
	output_dir = args.output
	np.linspace(0.1, 1.0, 10)

	if args.ml[0]:
		# Set model parameter for random Forest
		print("Random Forest (RF)...")
		param = {'max_features': 'auto',
				 'n_estimators': 2000,
				 'criterion': 'entropy',
				 'min_samples_leaf': 1}
		modelRF = RandomForestClassifier(**param)
		# Do cross-validation procedure with 10 folds
		r = crossvalidation(modelRF, df_new, 10)
		# Plot the AUC results
		# r contains acc, sens, spec, and results
		print_results(r[0], r[1], r[2], r[3], r[4], 'Random forest ROC curves', 'rf_roc.png', output_dir, 3)

	if args.ml[1]:
		# Specify model for SVM
		print("Support Vector Machines (SVM)...")
		modelSVM = svm.SVC(kernel='rbf', C=1, gamma=0.1, probability=True)
		# Do cross-validation procedure with 10 folds
		r = crossvalidation(modelSVM, df_new, 10)
		# Plot results
		print_results(r[0], r[1], r[2], r[3], r[4], 'SVM$(rbf kernel)$ $C=1$ $\gamma=0.1$ ROC curves', 'svm_roc.png', output_dir, 3)

	if args.ml[2]:
		# Specify model, default activation: relu
		print("Multilayer Perceptron (MLP)...")
		modelMLP = MLPClassifier(solver='adam', 
								 alpha=1e-5, 
								 hidden_layer_sizes=(5, 3), 
								 random_state=1, early_stopping=False)
		# Do cross-validation procedure with 10 folds
		r = crossvalidation(modelMLP, df_new, 10)
		# Plot results
		print_results(r[0], r[1], r[2], r[3], r[4], 'MLPClassifier ROC curves', 'mlp_roc.png', output_dir, 3)

	if args.ml[3]:
		# Specify model for Nearest Neighbor
		print("k-Nearest Neighbor (kNN)...")
		k = int(math.sqrt(df_new.shape[0]))
		modelKNN = KNeighborsClassifier(n_neighbors=k)
		# Do cross-validation procedure with 10 folds
		r = crossvalidation(modelKNN, df_new, 10)
		# Plot results
		print_results(r[0], r[1], r[2], r[3], r[4], 'KNNClassifier ROC curves', 'knn_roc.png', output_dir, 3)

	if args.ml[4]:
		# Specify model for Gaussian Process Classifier
		print("Gaussian Process Classifier (GPC)...")
		kernel = 1.0 * RBF(1.0)
		modelGPC = GaussianProcessClassifier(kernel=kernel)
		# Do cross-validation procedure with 10 folds
		r = crossvalidation(modelGPC, df_new, 10)
		# Plot results
		print_results(r[0], r[1], r[2], r[3], r[4], 'GPClassifier ROC curves', 'gpc_roc.png', output_dir, 3)


def run(args):
	## Connect to ChEMBL database
	print("Connecting to ChEMBL database...")
	targets = new_client.target
	compounds = new_client.molecule
	bioactivities = new_client.activity

	## Target Data
	print("Get target data...")
	uniprot_id = args.uid
	# Get target information from ChEMBL but restrict to specified values only
	target_Uniprot = targets.get(target_components__accession=uniprot_id).only('target_chembl_id', 'organism', 'pref_name', 'target_type')
	pd.DataFrame.from_records(target_Uniprot)
	target = target_Uniprot[0]
	chembl_id = target['target_chembl_id']

	## Bioactivity Data
	print("Calculating Bioactivity data...")
	bioact_df = bioactivity(chembl_id, bioactivities)

	## Compund Data
	print("Calculating Compound data...")
	output_df = compound(bioact_df, compounds)
	print(output_df.shape[0], "rows")

	## Rule of Five Filter
	print("Filtering with rule of five...")
	filtered_df = filter_rule_of_five(output_df)
	print(filtered_df.shape[0], "rows left")

	## Filtering for PAINS
	if args.pains:
		print("Filtering for PAINS...")
		df_new = pains(filtered_df)
		print(df_new.shape[0], "rows left")
	else:
		df_new = no_pains(filtered_df)
		
	## Machine Learning
	print("Starting Machine Learning algorithms...")
	ml_algorithms(df_new, args)

	print("Done.")

if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	args = define_args()
	run(args)
