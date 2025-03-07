import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score



class FileInfo:

	def __init__(self, sense_f='race'):
		if sense_f=='race':
			self.files = [['datasets/homelessData_notprevelig_080620.csv',   'reentered',            ['Gender', 'Ethnicity'],                                    {'PrimaryRace': (3, 5), 'treat': (1, 3)},                   ['PrimaryRace'],  1],
						  #['datasets/recidivism/Data_1980.csv',              'RECID',                ['TIME','FILE', 'FOLLOW'],                                 {},                                                          ['WHITE'],        1],
						  ['dataset/compas1.csv',                            'two_year_recid',       ['sex'],                                                    {},                                                          ['race'],         1],
						  ['datasets/adult/adult.csv',                       'income',               ['native-country', 'gender', 'exitage', 'fnlwgt'],          {'gender': ('Female', 'Male'), 'income': ('<=50K', '>50K')}, ['race'],         0],
						  ['dataset/communities2.csv',                       'ViolentCrimesPerPop',  [],                                                         {},                                                          ['race'],         1],
						  ['dataset/lawschool3.csv',                         'bar1',                 ['gender', 'age'],                                          {'fulltime': (2.0, 1.0)},                                    ['race'],         0],
						  ['dataset/student-mat.csv',                        'G3',                   [],                                                         {'sex': ('F', 'M')},                                         [None],           0],
						  ['datasets/german1.csv',                            'Creditable',          ['Age'],                                                    {},                                                          [None],           0]]
		elif sense_f=='gender':
			self.files = [['datasets/homelessData_notprevelig_080620.csv',   'reentered',            ['PrimaryRace', 'Ethnicity'],                               {'PrimaryRace': (3, 5), 'treat': (1, 3), 'Gender': (0,1)},   ['Gender'],  1],
						  ['dataset/compas1.csv',                            'two_year_recid',       ['race'],                                                   {'sex':(1, 0)},                                              ['sex'],     1],
						  ['datasets/adult/adult.csv',                       'income',               ['native-country', 'race', 'exitage', 'fnlwgt'],            {'gender': ('Female', 'Male'), 'income': ('<=50K', '>50K')}, ['gender'],  0],
						  ['dataset/communities2.csv',                       'ViolentCrimesPerPop',  [],                                                         {},                                                          [None],      1],
						  ['dataset/lawschool3.csv',                         'bar1',                 ['race', 'age'],                                            {'gender':(1, 0), 'fulltime': (2.0, 1.0)},                   ['gender'],  0],
						  ['dataset/student-mat.csv',                        'G3',                   [],                                                         {'sex': ('F', 'M')},                                         ['sex'],     0],
						  #['datasets/german1.csv',                           'Creditable',           ['Age'],                                                   {},                                                          ['Sex'],     0],]
						  ['datasets/german_age.csv',                         'Creditable',           [],                                                        {},                                                          ['Age'],     0]]
		else:
			print('type of sense feat: \"' + str(sense_f) + '\" given to FileInfo is not valid')
			exit(-1)
		self.f_save_names =  ['homeless',
							  'recidivism',
							  'adult',
							  'communities',
							  'lawschool',
							  'student',
							  'german']

		self.f_names = ['Homeless',
						'Recidivism',
						'Income',
						'Crime',
						'Law School',
						'Student',
						'Credit']

		self.ordered_feats = [['UnrelatedChildren', 'UnrelatedAdults', 'UnrelatedChildren', 'num_members', 'Children15_17',
							   'Children11_14', 'Children6_10', 'Children3_5', 'Children0_2', 'TotalChildren'],
							  [],
							  [],
							  ['LemasGangUnitDeploy', 'MedNumBR'],
							  ['fam_inc'],
							  ['healthStatus', 'Walc', 'Dalc', 'goout', 'freetime', 'famrel', 'failures', 'studytime', 'traveltime',
							   'Fedu', 'Medu'],
							  ['CheckingSize', 'Savings', 'EmploymentLength', 'InstallmentRate', 'HomeOwnedTime', 'NumCredits']]

		self.cata_feats = [['PriorResidence', 'HousingStatusAtEntry', ],
						   [],
						   [],
						   [],
						   [],
						   ['cluster'],
						   []]

		self.cont_feats = [[],
						   [],
						   [],
						   [],
						   [],
						   [],
						   []]


		if sense_f == 'gender':
			self.group_names = [['Female', 'Male'], ['Female', 'Male'], ['Female', 'Male'], ['Female', 'Male'],
					  		    ['Female', 'Male'], ['Female', 'Male'], ['Female', 'Male']]
		elif sense_f == 'race':
			self.group_names = [['Black', 'White'], ['Black', 'White'], ['Non-White', 'White'], ['None-White', 'White'],
					  		    ['Non-White', 'White'], [], []]

		self.invariant_feats = [['treat'], [], [], [], [], []]
		self.protected = [ff[4] for ff in self.files]

		self.one_hots_cols = {}
		self.norm_cols = {}
		self.all_columns_after_transform = []
		self.original_columns = []
		self.set_cata_cols = []

		self.last_cata_cols = []
		self.last_cont_cols = []
		self.last_ordered_cols = []

	def add_attributes_to_remove(self, attribute_list_for_each_dataset=[]):
		for i, lst in enumerate(attribute_list_for_each_dataset):
			self.files[i][2] += lst

	def prep(self, f_index, cata=False, features_to_keep_after_prep=None, groups=False, even_lables=False, scaling='0-1',
			 normalize_binary=False, one_hot_labels=False, verbose=False, remove_outliers=None, drop_treat=False, top_k='all', numpy=False, remove_sense=True) -> object:
		file_name_, target_name_, remove_cols_, bin_vals_, sense_feats_, rev_ = self.files[f_index]

		if not remove_sense:
			remove_cols_ = []
		##########################################################
		df = pd.read_csv(file_name_)
		for col in df.columns:
			if 'Unnamed' in col:
				df.drop(col, axis=1, inplace=True)
		if verbose:
			print('    original shape', df.shape)
		for col in bin_vals_.keys():
			df = df[(df[col] == bin_vals_[col][0]) | (df[col] == bin_vals_[col][1])]
			df[col] = pd.Series(np.array([0 if df[col][i] == bin_vals_[col][0] else 1 for i in df.index]),
								index=df.index)
		if drop_treat and 'treat' in df.columns:
			remove_cols_ += ['treat']
		print('removed features:', remove_cols_)
		if len(remove_cols_) > 0:
			df.drop(remove_cols_, axis=1, inplace=True)




		if even_lables and (0.55 < np.mean(df[target_name_]) or np.mean(df[target_name_]) < 0.45):
			maj_lab = 1
			if np.mean(df[target_name_]) < 0.5:
				maj_lab = 0
			df_majority = df[df[target_name_] == maj_lab]
			df_minority = df[df[target_name_] == 1 - maj_lab]

			maj_size = sum([1 for lab in df[target_name_] if lab==maj_lab])

			df_minority_upsampled = resample(df_minority,
											 replace=True,
											 n_samples=maj_size,
											 random_state=101)

			# Combine majority class with upsampled minority class
			df = pd.concat([df_majority, df_minority_upsampled])
			df = df.sample(frac=1, random_state=101)





		cata_cols, cont_cols = [], []
		cata_thresh = 7
		self.original_columns = list(df.columns)
		for col in df.columns:
			if str in [type(elem) for elem in df[col].unique()] or col in self.cata_feats:
				cata_cols.append(col)
			elif 2 < df[col].nunique() < cata_thresh and col not in self.ordered_feats[f_index]:
				cata_cols.append(col)
			elif df[col].nunique() >= cata_thresh or col in self.ordered_feats[f_index]:
				cont_cols.append(col)

		if remove_outliers is not None:
			for col in cata_cols + self.ordered_feats[f_index]:

				cnts = df[col].value_counts()
				cnts = list(zip(cnts.to_numpy(), np.array(cnts.index)))
				outliers = [v for cnt, v in cnts if cnt <= len(df[col])*remove_outliers]
				#print('  ', col)
				#print('    removing', outliers, 'from', list(set(df[col])))
				for outlier in outliers:
					df = df[df[col] != outlier]

			empty_cols = []
			for col in df.columns:
				if len(set(df[col].to_numpy())) == 1:
					empty_cols.append(col)

			print('dropping empty cols:', empty_cols)
			df.drop(empty_cols, axis=1, inplace=True)
			for col in empty_cols:
				if col in cata_cols:
					cata_cols.remove(col)
				if col in self.ordered_feats[f_index]:
					self.ordered_feats[f_index].remove(col)
				if col in cont_cols:
					cont_cols.remove(col)

		if cata:
			for col in cata_cols:
				if str in [type(elem) for elem in df[col].unique()]:
					print('Removed Column:', col, 'from dataset:', self.files[f_index])
					df.drop(col, axis=1, inplace=True)
				else:
					if scaling=='0-1':
						df[col] = (df[col] - df[col].min()) / float(df[col].max() - df[col].min())
					elif scaling=='normalize':
						df[col] = df[col] - np.mean(df[col])
						df[col] = df[col]/np.std(df[col])
		else:
			df = pd.get_dummies(df, columns=cata_cols)
		for col in cont_cols:
			# df[col] = (df[col] - df[col].min()) / float(df[col].max() - df[col].min())
			if scaling == '0-1':
				df[col] = (df[col] - df[col].min()) / float(df[col].max() - df[col].min())
			elif scaling=='normalize':
				df[col] = df[col] - np.mean(df[col])
				df[col] = df[col] / np.std(df[col])
			elif scaling=='both':
				df[col] = df[col] - np.mean(df[col])
				df[col] = df[col] / np.std(df[col])
				df[col] = (df[col] - df[col].min()) / float(df[col].max() - df[col].min())

			else:
				print('ERROR: scaling=',scaling, 'is not a valid parameter')
				return None

			  
		if features_to_keep_after_prep is not None:
			for col in df.columns:
				if col not in features_to_keep_after_prep:
					df.drop(col, axis=1, inplace=True)
		if 'treat' in df.columns.to_list():
			col_list = list(df.columns)
			col_list.remove('treat')
			df = df[['treat'] + col_list]
		if normalize_binary:
			for col in df.columns:
				if col != 'treat' and col != self.protected[f_index][0] and len(df[col].unique()) <= 2:
					df[col] = df[col] - np.mean(df[col])
					df[col] = df[col] / np.std(df[col])

		self.last_cata_cols = set(cata_cols + self.cata_feats[f_index])
		self.last_cont_cols = set(cont_cols + self.ordered_feats[f_index])

		target = df[target_name_]
		if rev_:
			target = 1 - target
		if one_hot_labels:
			target = np.array([[0, 1] if yy == 0 else [1, 0] for yy in target])
		df.drop(target_name_, inplace=True, axis=1)

		del_cols = [col for col in self.original_columns if not any(col in Xcol for Xcol in df.columns)]
		for col in del_cols:
			self.original_columns.remove(col)

		if type(top_k) == int:
			df = self.get_top_k_predictive_features(df, target, top_k, f_index)


		del_cont_cols = []
		del_cata_cols = []
		for col in self.last_cont_cols:
			if not any(col in df_col for df_col in df.columns):
				del_cont_cols.append(col)
		for col in self.last_cata_cols:
			if not any(col in df_col for df_col in df.columns):
				del_cata_cols.append(col)

		for col in del_cont_cols:
			self.last_cont_cols.remove(col)
		for col in del_cata_cols:
			self.last_cata_cols.remove(col)


		if groups:
			A = df[self.protected[f_index][0]]
			df.drop(self.protected[f_index][0], inplace=True, axis=1)
			if numpy:
				if one_hot_labels:
					return df.to_numpy(), target, A.to_numpy()
				else:
					return df.to_numpy(), target.to_numpy(), A.to_numpy()
			return df, target, A
		if numpy:
			return df.to_numpy(), target.to_numpy()
		return df, target

	def get_top_k_predictive_features(self, X, y, k, f_index):
		if len(self.original_columns) < 1:
			print('self.original_columns is empty')
			exit()
		cols_and_scores = []

		for o_col in self.original_columns:
			cols = [col for col in X.columns if o_col in col]
			if len(cols) == 0:
				print(o_col, 'is empty')
			df_to_use = X[cols]
			acc = accuracy_score(y, GradientBoostingClassifier().fit(df_to_use, y).predict(df_to_use))
			cols_and_scores.append((cols, acc))
		cols_and_scores = sorted(cols_and_scores, key=lambda x: x[1])[::-1]
		cols = [col for cols, score in cols_and_scores[:k] for col in cols]
		print([(cols[0], round(score, 2)) for cols, score in cols_and_scores[:k]])
		if self.protected[f_index][0] not in cols:
			return X[cols + [self.protected[f_index][0]]]
		else:
			cols = [col for cols, score in cols_and_scores[:k+1] for col in cols]
			return X[cols]





class FairnessFunctions:
	@staticmethod
	def PR(p, y):
		return np.mean(p)

	@staticmethod
	def FPR(p, y):
		fp = [1 for pp, yy in zip(p, y) if pp==1 and yy==0]
		return np.sum(fp)/np.float(np.sum(1-y))

	@staticmethod
	def TPR(p, y):
		tp = [1 for pp, yy in zip(p, y) if pp==1 and yy==1]
		return np.sum(tp) / np.float(np.sum(y))

	@staticmethod
	def EPR(p, y):
		return np.mean(p)

	@staticmethod
	def EFPR(p, y):
		fp = np.sum([pp for pp, yy in zip(p, y) if yy==0])
		return fp/np.float(np.sum(1-y))

	@staticmethod
	def ETPR(p, y):
		tp = np.sum([pp for pp, yy in zip(p, y) if yy==1])
		return tp/np.float(np.sum(y))

	@staticmethod
	def EACC(p, y):
		return np.mean(p*y + (1-p)*(1-y))

	@staticmethod
	def ACC(p, y):
		return np.mean(p*y + (1-p)*(1-y))


	@staticmethod
	def Udiff(clf, X, y, A, M, abs_val=True, prob=False, fp=False, pred_vals=None, proba_vals=None):
		g0 = np.argwhere(A==0).flatten()
		g1 = np.argwhere(A==1).flatten()
		pred = None
		if prob:
			if proba_vals is not None:
				pred = proba_vals
			elif fp:
				pred = clf.predict_proba(X, A)[:,1]
			else:
				pred = clf.predict_proba(X)[:,1]
		else:
			if pred_vals is not None:
				pred = pred_vals
			elif fp:
				pred = clf.predict(X, A)
			else:
				pred = clf.predict(X)

		if abs_val:
			return abs(M(pred[g1], y[g1]) - M(pred[g0], y[g0]))
		else:
			return M(pred[g1], y[g1]) - M(pred[g0], y[g0])

	@staticmethod
	def _Udiff(pred, y, A, M, abs_val=True, prob=False):
		g0 = np.argwhere(A == 0).flatten()
		g1 = np.argwhere(A == 1).flatten()
		if abs_val:
			return abs(M(pred[g1], y[g1]) - M(pred[g0], y[g0]))
		else:
			return M(pred[g1], y[g1]) - M(pred[g0], y[g0])

	@staticmethod
	def get_costs(M_name, y, A):
		c = np.zeros((2, 2, 2))
		if M_name == 'PR' or M_name == 'EPR':
			c[0, :, 1] = 1.0/len([1 for aa in A if aa==0])
			c[1, :, 1] = 1.0/len([1 for aa in A if aa==1])
		elif M_name == 'TPR' or M_name == 'ETPR':
			c[0, 1, 1] = 1.0/max(1, len([1 for yy, aa in zip(y, A) if yy==1 and aa==0]))
			c[1, 1, 1] = 1.0/max(1, len([1 for yy, aa in zip(y, A) if yy==1 and aa==1]))
		elif M_name == 'FPR' or M_name == 'EFPR':
			c[0, 0, 1] = 1.0/max(1, len([1 for yy, aa in zip(y, A) if yy==0 and aa==0]))
			c[1, 0, 1] = 1.0/max(1, len([1 for yy, aa in zip(y, A) if yy==0 and aa==1]))
		elif M_name == 'ERR' or M_name == 'EERR':
			c[0, 0, 1] = 1 / len([1 for yy, aa in zip(y, A) if yy == 0 and aa == 0])
			c[0, 1, 0] = 1 / len([1 for yy, aa in zip(y, A) if yy == 1 and aa == 0])
			c[1, 0, 1] = 1 / len([1 for yy, aa in zip(y, A) if yy == 0 and aa == 1])
			c[1, 1, 0] = 1 / len([1 for yy, aa in zip(y, A) if yy == 1 and aa == 1])
		else:
			print(M_name, 'is not a valid metric')
		return c









