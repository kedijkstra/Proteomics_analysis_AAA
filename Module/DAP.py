"""
DAP.py

This module contains functionalities for the identification of 
differentially abundant proteins.

Author: Keimpe Dijkstra
Date: 27-6-2025
"""

#LIBRARIES
import pandas as pd
import scipy.stats as ss
from scipy.spatial import distance
import statsmodels.api as sm
import statsmodels.stats.multitest as smsm
import matplotlib.pyplot as plt
import statistics
import math
import numpy as np
from sklearn.impute import KNNImputer
import seaborn as sns
import sklearn.metrics as sklm
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
import umap.umap_ as UMAP
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from sklearn.preprocessing import normalize
import numbers
from Pipelines.vsn import VSN
from sklearn.ensemble import *
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
import qnorm
from sklearn import svm
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from scipy.cluster.hierarchy import linkage
import matplotlib.patches as mpatches


class DifferentiallyAbundantProteins():
    '''
    This class contains functionalities for the identification of 
    differentially abundant proteins.

    
    Attributes
    ------------
    df : pandas dataframe
        Wide format pandas dataframe with quantification and other data.
    id : str
        String containing the name of the column with the sample identification ID's
    group : str
        String containing the name of the column with the groupings (ex. healthy/treatment)
    proteins : list
        List of strings containing the column names of proteins
    groups : list
        List of unique groups derived from group column
    log_fold_thres : float
        Threshold for log fold change to be considered an DAP
    '''
    
    def __init__(self, dataframe, ID_column, group_column, proteins):
        '''
        Init function of DifferentiallyAbundantProteins class.
        
        PARAMETERS
        ------------
        dataframe : pandas dataframe
            Wide format pandas dataframe with quantification and other data.
        ID_column : str
            String containing the name of the column with the sample identification ID's
        group_column : str
            String containing the name of the column with the groupings (ex. healthy/treatment)
        proteins : list
            List of strings containing the column names of proteins
        '''
        self.df = dataframe.copy() #Pandas dataframe object 
        self.id = ID_column #Sample ID's
        self.group = group_column #Column containing the groups assigned (e.g. patient/control), needs to be 2
        self.proteins = proteins #List of columns containing protein quantifications

        self.groups = self.df[self.group].unique() #List of groups
        self.log_fold_thres = 0.5 #Threshold for log fold change to be considered an DAP

    #GETTERS AND SETTERS
    
    def get_data(self):
        '''Retrieve the data'''
        return self.df
    
    def set_data(self, data):
        '''
        Set the data

        PARAMETERS
        ------------
        data : pandas dataframe
            Wide format pandas dataframe with quantification and other data.
        '''
        self.df = data.copy()
    
    def get_proteins(self):
        '''Retrieve list of proteins'''
        return self.proteins
    
    def set_proteins(self, proteins):
        '''
        Set the list of proteins

        PARAMETERS
        ------------
        proteins : list
            List of strings containing the column names of proteins
        '''
        self.proteins = proteins

    def get_group(self):
        '''Retrieve the group column identifier'''
        return self.group
    
    def get_groups(self):
        '''Retrieve the unique groups'''
        return self.groups

    def set_groups(self, group):
        '''
        Set the group and subsequent groups, 
        make sure column is present in data.

        PARAMETERS
        ------------
        group : str
            String containing the name of the column with the groupings (ex. healthy/treatment)
        '''
        self.group = group
        self.groups = self.df[self.group].unique()

    def get_id(self):
        '''Retrieve the sample id column identifier'''
        return self.id
    
    def set_id(self, id):
        '''
        Set the id column identifier

        PARAMETERS
        ------------
        id : str
            String containing the name of the column with the sample identification ID's
        '''
        self.id = id

    def get_log_fold_thres(self):
        '''Retrieve the log fold change threshold'''
        return self.log_fold_thres
    
    def set_log_fold_thres(self, lft):
        '''
        Set the id column identifier

        PARAMETERS
        ------------
        lft : float
            Threshold for log fold change to be considered an DAP
        '''
        self.log_fold_thres = lft
    
    #PREPROCESSING FUNCTIONS

    def missing(self, proteins=False, percentage=False):
        '''
        This function determines the amount of missing data per protein.
        
        PARAMETERS
        ------------
        percentage : boolean
            if true return missing value as percentage of total
            else absolute number (default = False)
        proteins : list 
            list of proteins to use (default = self.proteins)

        RETURNS
        ------------
        results : pandas dataframe 
            pandas dataframe  containing protein and missing count columns
        '''
        if type(proteins) != list:
            proteins = self.proteins

        results = pd.DataFrame(columns=["protein", "n_missing"]) 

        for p in proteins:
            results.loc[len(results)] = [p, self.df[p].isna().sum()]

        if percentage:
            results["n_missing"] = results["n_missing"].div(len(self.df)).multiply(100)

        return results
    
    def average_quant(self, method="mean", proteins=False, na_zero=False):
        '''
        This function determines the average amount of protein quantified.
        
        PARAMETERS
        ------------
        method : str
            Method of determining average {mean, median} (default = mean)
        na_zero : boolean
            include missing values as 0 (default = False)
        proteins : list 
            list of proteins to use (default = self.proteins)

        RETURNS
        ------------
        results : pandas dataframe 
            pandas dataframe containing protein and average quantification
        '''
        if type(proteins) != list:
            proteins = self.proteins

        results = pd.DataFrame(columns=["protein", "avg_quant"]) 

        for p in proteins:
            t = list(self.df[p])

            if na_zero:
                t = [0 if math.isnan(i) else i for i in t]
            else:
                t = [i for i in t if not math.isnan(i)]

            if method == "mean":
                results.loc[len(results)] = [p, statistics.mean(t)]
            elif method == "median":
                results.loc[len(results)] = [p, statistics.median(t)]
            else:
                print("Method not recognized: ", method)
                break

        return results
            
    def avg_filtering(self, threshold, proteins=False, method="mean", na_zero=False):
        '''
        This function determines the average amount of protein quantified and filters the
        dataset on an arbitrary threshold thereof.
        
        PARAMETERS
        ------------
        threshold : float
            integer/float which serves as the cut-off value for proteins to delete
        method : str
            Method of determining average {mean, median} (default = mean)
        na_zero : boolean
            include missing values as 0 (default = False)
        proteins : list 
            list of proteins to use (default = self.proteins)

        DEPENDENCIES
        ------------
        function : average_quant
        '''
        if type(proteins) != list:
            proteins = self.proteins

        proteins_filtered = self.average_quant(method=method, 
                                                na_zero=na_zero,
                                                proteins=proteins)
        proteins_filtered = proteins_filtered[proteins_filtered["avg_quant"] <= threshold]
        self.df = self.df.drop(columns=proteins_filtered["protein"].tolist())
        self.proteins = [p for p in self.proteins if p not in proteins_filtered["protein"].tolist()]
        
    def missing_filtering(self, threshold, proteins=False, percentage=False):
        '''
        This function determines the amount of missing data per protein and filters the
        dataset on an arbitrary threshold thereof. This only alters the list of proteins.
        
        PARAMETERS
        ------------
        threshold : float
            integer/float which serves as the cut-off value for proteins to delete
        percentage : boolean
            if true return interpret threshold as percentage of total
            else absolute number (default = False)
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        proteins_missing = self.missing(proteins=proteins, percentage=percentage)
        proteins_missing = proteins_missing[proteins_missing["n_missing"] > threshold]
        self.proteins = [p for p in self.proteins if p not in proteins_missing["protein"].tolist()]
        
    def log_transform(self, proteins=False):
        '''
        This function log-transforms data from selected proteins.
        
        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        for p in proteins:
            self.df[p] = np.log2(self.df[p])

    def exponentiate(self, base="2", proteins=False):
        '''
        This function exponentiates data from selected proteins.

        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        for p in proteins:
            for n in range(len(self.df[p])):
                self.df.at[n, p] = 2 ** self.df.at[n, p]

    def nan_to_zero(self, proteins=False):
        '''
        This function replaces non-existing values with zero's for 
        selected proteins.
        
        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        for p in proteins:
            self.df[p] = [0 if math.isnan(float(i)) else i for i in self.df[p]]

    def laplace(self, value=1, proteins=False):
        '''
        This function replaces non-existing values with a given value for 
        selected proteins.
        
        PARAMETERS
        ------------
        value : float
            value to replace nan's with (default = 1)
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        for p in proteins:
            self.df[p] = [value if math.isnan(float(i)) else i for i in self.df[p]]
    
    def avg_imputation(self, proteins=False):
        '''
        This function imputes non-existing values with the protein average.
        
        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        for p in proteins:
            avg = np.nanmean(self.df[p])
            self.df[p] = [avg if math.isnan(i) else i for i in self.df[p]]

    def knn_imputation(self, n_neighbors=10 ,proteins=False):
        '''
        This function imputes non-existing values with the a value determined 
        by a K-NN imputer.
        
        PARAMETERS
        ------------
        n_neighbors : int
            number of neightbors to train the k-nn imputer on
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        imputer = KNNImputer(n_neighbors=n_neighbors)
        self.df[proteins] = imputer.fit_transform(self.df[proteins])

    def miss_forest_imputation(self, n_iter ,proteins=False):#TODO: Check functionality
        '''
        This function performs miss random forest imputation

        PARAMETERS
        ------------
        n_iter : int
            number of iterations
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        #Imputed dataset with means
        avg_df = pd.DataFrame()
        for p in proteins:
            avg = np.nanmean(self.df[p])
            avg_df[p] = [avg if math.isnan(i) else i for i in self.df[p]]

        #Mask missing data
        mask = self.df.isna()

        for i in range(n_iter):
            for p in proteins:
                #Select data from avg dataset on first run
                if i == 0:
                    train = avg_df[~mask[p]]
                    test = avg_df[mask[p]]
                elif i >0:
                    train = self.df[~mask[p]]
                    test = self.df[mask[p]]
                
                if len(test) > 0: #Check if there are missing values
                    #Train the rf model
                    tproteins = proteins.copy()
                    tproteins.remove(p)
                    rf = RandomForestRegressor(n_jobs=-1)
                    rf.fit(train[tproteins], train[p])
                    pred = rf.predict(test[tproteins])

                    #Assign values
                    self.df.loc[mask[p], p] = pred

    def maximum_likelihood(self, cutoff, proteins=False):
        '''
        Estimates the parameters (mean and standard deviation) of a normal distribution 
        for each specified protein using maximum likelihood estimation (MLE), based on 
        observed values in the object's dataframe.

        Parameters:
        ----------
        cutoff : float 
            threshold value used in the likelihood calculation
        proteins : list 
            list of proteins to use (default = self.proteins)

        Returns:
        -------
        means : list
            list of the estimated means of the normal distributions for each protein.
        sds : list
            list of the estimated standard deviations of the normal distributions for each protein.
        '''
        if type(proteins) != list:
            proteins = self.proteins

        means = []
        sds = []
        for p in proteins:
            obs_vals = self.df[p].tolist()
            obs_vals = [x for x in obs_vals if str(x) != 'nan']

            #Initial values
            mean = statistics.mean(obs_vals)
            sd = np.std(obs_vals)

            result = minimize(self.neg_log_likelihood, [mean, sd], args=(cutoff, obs_vals),bounds=[(None, None), (1e-6, None)])
            est_mean, est_sd = result.x

            means.append(est_mean)
            sds.append(est_sd)

        return means, sds
                
    def median_normalize(self, proteins=False):
        """
        This function median normalizes selected proteins.
        
        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)
        """
        if type(proteins) != list:
            proteins = self.proteins

        medians = self.df[proteins].median()
        self.df[proteins] = self.df[proteins] - medians

    def skl_normalize(self, proteins=False):
        '''
        This function normalizes data of selected proteins using the sklearn nromalization function.

        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        sub_df = self.df[proteins].copy()

        # Normalize each row, ignoring NaNs
        self.df[proteins] = sub_df.apply(self.normalize_row, axis=1)

    def normalize_row(self, row):
        '''
        Helper function of the skl_normalize fucntion, performs normalization for a given row.

        PARAMETERS
        ------------
        row : int
            integer indicating the row position
        '''
        mask = ~row.isna()
        if mask.sum() == 0:
            return row  # nothing to normalize
        normalized_vals = normalize([row[mask].values])[0]
        row.loc[mask] = normalized_vals
        return row
    
    def quantile_normalization(self, proteins=False):
        '''
        This function performes quantile normalization for 
        selected proteins

        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        self.df[proteins] = qnorm.quantile_normalize(self.df[proteins], axis=0)

    def quantile_normalize_group(self, proteins=False):
        '''
        This function performend quantile normalization function for 
        selected proteins based on groups

        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)

        Notes
        ------------
        - Only a number of 2 groups is supported
        '''
        if type(proteins) != list:
            proteins = self.proteins
        one = self.df[self.df[self.group] == self.groups[0]]
        two = self.df[self.df[self.group] == self.groups[1]]

        one = qnorm.quantile_normalize(one[proteins], axis=1)
        two = qnorm.quantile_normalize(two[proteins], axis=1)
        self.set_data(pd.concat([one, two], ignore_index=True))
        
    def median_ratio_normalization(self, proteins=False):
        '''
        This function corrects the data using median-ratio
        normalization.
        
        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        #Calculate geometric means
        means = {}
        for p in self.proteins:
            #means[p] = statistics.geometric_mean(self.df[p])
            values = self.df[p].dropna()
            means[p] = np.exp(np.nanmean(np.log(values)))

        #Calculate fold changes
        fold_change = {}
        x=0
        for s in self.df[self.id]: #For each sample
            fold_change[s] = {}
            for p in self.proteins: #For each protein for the sample 
                fold_change[s][p] = self.df[p].iloc[x] / means[p]
            x+=1

        #Calculate size factors
        size_factors = {} #size factors (medians) for samples
        for s in fold_change:
            #size_factors[s] = statistics.median(list(fold_change[s].values()))
            fc_values = [v for v in fold_change[s].values() if not pd.isna(v)]
            size_factors[s] = np.nanmedian(fc_values) if fc_values else np.nan

        x=0
        #Correct data
        for s in self.df[self.id]:
            for p in proteins:
                if not pd.isna(self.df.iloc[x][p]):
                    self.df.iloc[x, self.df.columns.get_loc(p)] = self.df.iloc[x][p] / size_factors[s]
            x+=1
                      
    def total_intensity_normalization(self, method="mean" ,proteins=False):
        '''
        This function corrects the data using total intensity
        normalization.
        
        PARAMETERS
        ------------
        method : str
            Method of determining average {mean, median} (default = mean)
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        #Calculate the intensities for each sample
        intensities = {}
        for s in self.df[self.id]:
            intensities[s] = (self.df.loc[self.df[self.id]==s,proteins].sum(axis=1, numeric_only=True).item())

        if method == "mean":
            m = statistics.mean(list(intensities.values()))
        if method == "median":
            m = statistics.median(list(intensities.values()))

        #Calculate the  normalization factors
        for s in intensities:
            intensities[s] = m/intensities[s]

        #Correct individual proteins with factor; 
        for s in self.df[self.id]:
            for p in proteins:
                self.df.loc[self.df[self.id]==s, p] = self.df.loc[self.df[self.id]==s, p] * intensities[s]

    def ma_transform(self, x, y, log_transformed=True):
        '''
        Calculate M (log ratio) and A (mean average) for paired data.

        PARAMETERS
        ------------
        x : list
            list of numeric data
        y : list
            list of numeric data (paired data in order with respect to y)
        log_transformend : boolean 
            indicates if data has already been log transformend

        Returns:
        -------
        m : list
            list of log ratio's
        a : list
            list of mean averages
        '''
        m = []
        a = []

        if not log_transformed:
            x = np.log2(x)
            y = np.log2(y)

        for i in range(len(x)):
            if not np.isnan(x[i]) and not np.isnan(y[i]):
                m.append(x[i] - y[i])
                a.append(0.5*(x[i] + y[i]))

        return m, a

    def loess(self, m, a):
        '''
        This function performs the statmodels loess function with:
        * frac = 0.7
        * return_sorted = False

        PARAMETERS
        ------------
        m : list
            endog argument 
        a : list
            exog argument
        '''
        return sm.nonparametric.lowess(m, a, frac=0.7, return_sorted=False)
    
    def cyclic_loess(self, n_iter=3, proteins=False):
        '''
        This function performes a cyclic lowes based on a reference set containing median values.

        PARAMETERS
        ------------
        n_iter : int 
            indicates the number of iterations (default = 3)
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        medians = [self.df[p].median() for p in proteins]

        x = 0
        while x < n_iter:
            for s in self.df[self.id]:
                
                m, a = self.ma_transform(self.df[self.df[self.id] == s][proteins].values.tolist()[0], medians)
                predicted_m = self.loess(m, a)
                
                for x in range(len(m)):
                    corrected_m = m[x] - predicted_m[x]
                    rown1 = self.df.index[self.df[self.id] == s].tolist()[0]
                    self.df.at[rown1,proteins[x]] = a[x] + corrected_m / 2
            x+=1

    def variance_stabilizing_normalization(self, tol= 1e-6, max_iter=1000, proteins=False):
        '''
        This function executes the VSN function (from stmball on github)

        PARAMETERS
        ------------
        tol : float (optional) 
            Tolerance for gradient descent to stop (default = 1e-6)
        max_iter : int (optional)
            Max number of iterations before gradient descent stops. (default = 1000)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        vsn = VSN()
        vsn.fit(self.df[proteins].transpose().to_numpy(), tol=tol, max_iter=max_iter)
        res = vsn.transform(self.df[proteins].transpose().to_numpy())
        self.df[proteins] =  pd.DataFrame(res).transpose()

    def linear_regression_normalization(self, diag_plot=False, proteins=False):
        '''
        This function performens linear regression normalization on selected proteins

        PARAMETERS
        ------------
        diag_plot : boolean 
            indicates if diagnostic plot should be constructed (default = false)
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        medians = [self.df[p].median() for p in proteins]

        for s in range(len(self.df[self.id])):
            normalized_vals = []
            slope, intercept, r, p, std_err = ss.linregress(medians, self.df.loc[s, proteins].tolist())
            for n in medians:
                normalized_vals.append(self.linear_calc(slope=slope, intercept=intercept, x=n))
            
            if diag_plot:
                fig, ax = plt.subplots()
                ax.scatter(medians, self.df.loc[s, proteins].tolist())
                lin_model = list(map(lambda xi: self.linear_calc(xi, slope, intercept), medians))
                plt.plot(medians, lin_model, color="k")
                ax.set_title(self.df[self.id][s])
                ax.set_xlabel("median")
                ax.set_ylabel("sample value")
                plt.show()

            for n in range(len(proteins)):
                self.df.at[s,proteins[n]] = normalized_vals[n]
            
    def ma_linear_regression_normalization(self, n_iter=3, proteins=False):
        '''
        This function performens ma linear regression normalization on selected proteins in a cyclical manner

        PARAMETERS
        ------------
        n_iter : int 
            indicates the number of iterations (default = 3)
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        medians = [self.df[p].median() for p in proteins]

        x = 0
        while x < n_iter:
            for s in self.df[self.id]:
                
                m, a = self.ma_transform(self.df[self.df[self.id] == s][proteins].values.tolist()[0], medians)
                
                slope, intercept, r, p, std_err = ss.linregress(a, m)
                
                for x in range(len(m)):
                    predicted_m = self.linear_calc(slope=slope, intercept=intercept, x=m[x])
                    corrected_m = m[x] - predicted_m

                    rown1 = self.df.index[self.df[self.id] == s].tolist()[0]
            
                    self.df.at[rown1,proteins[x]] = a[x] + corrected_m / 2
            
            x+=1

    def fold_change(self, control_group, treatment_group ,proteins=False, logged=True, log=False):
        '''
        This function calculates the fold change value for
        selected proteins. Only executable with n group = 2.
        
        PARAMETERS
        ------------
        logged : boolean 
            indicates if data is log transformed
        log : boolean
            indicates if fold changes need to be log transformend
        control_group : str 
            group name of controls
        treatment_group : str 
            group name of treatment
        proteins : list 
            list of proteins to use (default = self.proteins)

        RETURNS
        ------------
        fold_change : nested dictionary 
            proteins and their fold change for all proteins
        '''
        if type(proteins) != list:
            proteins = self.proteins

        spm = self.sample_protein_map(proteins=proteins)

        fold_change = {}

        for p in spm:
            if logged:
                if log:
                    fold_change[p] = statistics.mean(spm[p][treatment_group]) - statistics.mean(spm[p][control_group])
                else:
                    fold_change[p] = statistics.mean(spm[p][treatment_group]) / statistics.mean(spm[p][control_group])
            
            else:
                if log:
                    fold_change[p] = np.log2(statistics.mean(spm[p][treatment_group]) / statistics.mean(spm[p][control_group]))
                else:
                    fold_change[p] = statistics.mean(spm[p][treatment_group]) / statistics.mean(spm[p][control_group])
                
        return fold_change
    
    #MATH FUNCTIONS

    def euclidian_distance_matrix(self, proteins=False):
        '''
        This function creates a distance matrix for all protein 
        values based on the euclidean distance metric.

        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)

        RETURNS
        ------------
        distances : matrix
            euclidean distance matrix
        '''
        if type(proteins) != list:
            proteins = self.proteins

        return sklm.euclidean_distances(self.df[proteins])
    
    def manhattan_distance_matrix(self, proteins=False):
        '''
        This function creates a distance matrix for all protein 
        values based on the manhattan distance metric.

        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)

        RETURNS
        ------------
        distances : matrix
            manhattan distance matrix
        '''
        if type(proteins) != list:
            proteins = self.proteins

        mdm = pd.DataFrame(columns=self.df[self.id], 
                           index=self.df[self.id])
        
        for s1 in self.df[self.id]:
            for s2 in self.df[self.id]:
                mdm.loc[s2, s1] = distance.cityblock(self.df.loc[self.df[self.id]==s1, proteins].values.flatten().tolist(), 
                                                     self.df.loc[self.df[self.id]==s2, proteins].values.flatten().tolist())

        return mdm.to_numpy().astype(np.float64)
    
    def poisson_distance_matrix(self, proteins=False):
        '''
        This function creates a distance matrix for all protein 
        values based on the poisson distance metric.

        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)

        RETURNS
        ------------
        distances : matrix
            poisson distance matrix
        '''
        if type(proteins) != list:
            proteins = self.proteins

        pdm = pd.DataFrame(columns=self.df[self.id], 
                           index=self.df[self.id])
        
        for s1 in self.df[self.id]:
            for s2 in self.df[self.id]:
                #Compute square roots of values
                sqrts1 = [math.sqrt(i) for i in self.df.loc[self.df[self.id]==s1,proteins].values.flatten().tolist()]
                sqrts2 = [math.sqrt(i) for i in self.df.loc[self.df[self.id]==s2,proteins].values.flatten().tolist()]         

                #Compute squared differences
                sqd_diff = []
                for n in range(len(proteins)):
                    sqd_diff.append((sqrts1[n] - sqrts2[n])**2)
                
                #Compute the square root of summation
                pdm.loc[s2, s1] = math.sqrt(sum(sqd_diff)) 
        
        return pdm.to_numpy().astype(np.float64)
    
    def multi_dimensional_scaling(self, distance_matrix):
        '''
        This function creates a multi dimensional scaling embedding
        based on a distance matrix, computes 2 components. 
        
        PARAMETERS
        ------------
        distance_matrix : matrix
            a distance matrix

        RETURNS
        ------------
        embedding : MDS object 
            fit-transformed by distance matrix.
        '''
        embedding = MDS(n_components=2,
                        dissimilarity = "precomputed")
        
        return embedding.fit_transform(distance_matrix)
    
    def pca(self, proteins=False):
        '''
        This function calculates the principal components of 
        given proteins. Before running, it is recommended to log
        transform and normalize the data.
        
        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)

        RETURNS
        ------------
        pca_res_f : PCA object
            fitted sklearn PCA object
        pca_res_ft : dataframe
            fit-transformed sklearn PCA results with number of components = 2 
        '''
        if type(proteins) != list:
            proteins = self.proteins

        pca_res = PCA()
        pca_res_f = pca_res.fit(self.df[proteins])

        pca_res = PCA(n_components=2)
        pca_res_ft = pca_res.fit_transform(self.df[proteins])

        return pca_res_f, pca_res_ft
    
    def tsne(self, distance_matrix, perplexity=30):
        '''
        This function creates a T-distributed Stochastic Neighbor Embedding
        based on a given distance matrix. Init is set to random.

        PARAMETERS
        ------------
        distance_matrix : matrix
            a distance matrix
        perplexity : int
            must be less than n samples (default = 30)
        
        RETURNS
        ------------
        embedding : TSNE sklearn object 
            fit-transformed by distance matrix.
        '''
        embedding = TSNE(metric="precomputed",
                         perplexity=perplexity,
                         init="random")
        return embedding.fit_transform(distance_matrix)
    
    def umap_2d(self, proteins=False):
        '''
        This function creates a Uniform manifold approximation and projection Embedding
        based on a selected proteins. Init is set to random.

        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)
        
        RETURNS
        ------------
        results : umap.umap object 
            fit-transformed by raw data
        '''
        if type(proteins) != list:
            proteins = self.proteins

        umap_2d_res = UMAP.UMAP(n_components=2,
                           init="random",
                           random_state=0)
        return umap_2d_res.fit_transform(self.df[proteins]).astype(np.float64)
    
    def pearson_correlation(self, proteins=False):
        '''
        This function calculates the pearson correlations for 
        given proteins from different samples.

        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)
        
        RETURNS
        ------------
        pc : dataframe 
            contains pearson correlation values
        pc_pvals : dataframe 
            contains p-values from correlation 
        '''
        if type(proteins) != list:
            proteins = self.proteins

        pc = pd.DataFrame(columns=self.df[self.id], 
                           index=self.df[self.id])
        pc_pvals = pd.DataFrame(columns=self.df[self.id], 
                           index=self.df[self.id])

        for s1 in self.df[self.id]:
            for s2 in self.df[self.id]:
                #Compute pearon correlation of samples
                x = [i for i in self.df.loc[self.df[self.id]==s1,proteins].values.flatten().tolist()]
                y = [i for i in self.df.loc[self.df[self.id]==s2,proteins].values.flatten().tolist()]     
                pearson_res = ss.pearsonr(x, y)

                pc.loc[s1, s2] = pearson_res[0]
                pc_pvals.loc[s1, s2] = pearson_res[1]
        return pc.to_numpy().astype(np.float64), pc_pvals.to_numpy().astype(np.float64)
    
    def missing_abundance_pearson(self, proteins=False):
        '''
        This function calculates the pearson correlations between missing
        and protein abundance.

        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)
        
        RETURNS
        ------------
        pearson : scipy.stats pearsonr object
        '''
        if type(proteins) != list:
            proteins = self.proteins

        mis = self.missing(proteins=proteins)["n_missing"].tolist()
        sums = [self.df[p].sum() for p in proteins]
        return ss.pearsonr(mis, sums)

    #STATISTICS FUNCTIONS

    def protein_batch_ttest(self, proteins=False, sign_only=False, alpha=0.05, multiple_correction="bh"):
        '''
        This function calculates a t-test for selected proteins
        based on the current groups. Only executable with n group = 2.

        PARAMETERS
        ------------
        multiple_correction : str
            {bh (Benjamini/Hochberg), bon (bonferroni)} (default = bh)
        sign_only : boolean 
            only return significant values (default = False)
        alpha : float
            threshold for significance (default = 0.05)
        proteins : list 
            list of proteins to use (default = self.proteins)

        RETURNS
        ------------
        pvals : dictionary
            proteins as key and pvalue as value
        '''
        if type(proteins) != list:
            proteins = self.proteins

        spm = self.sample_protein_map(proteins=proteins)
        
        #testing !TWO GROUPS ONLY!
        pvals = {}
        for p in proteins:
            res = ss.ttest_ind(spm[p][self.groups[0]],
                         spm[p][self.groups[1]], nan_policy="omit")
            pvals[p] = res.pvalue

        to_pop = []
        for protein, p in pvals.items():
            if (not isinstance(p, numbers.Number)) or np.isnan(p) or p < 0 or p > 1:
                print("protein ", protein, " with value ", p, " was omitted")
                to_pop.append(protein)
        for i in to_pop:
            pvals.pop(i)

        if multiple_correction == "bh":
            fdc_pval = ss.false_discovery_control(list(pvals.values()))
            x=0
            for p in pvals:
                pvals[p] = fdc_pval[x]
                x+=1

        if multiple_correction == "bon":
            reject, pvalsbon, alphasidak, alphabonf = smsm.multipletests(list(pvals.values()),
                                        alpha=alpha,
                                        method="bonferroni")
            x=0
            for p in pvals:
                pvals[p] = pvalsbon[x]
                x+=1

        if sign_only:
            pvals = {p: v for p, v in pvals.items() if v <= alpha}

        return pvals
    
    def protein_batch_anova(self, multiple_correction="bh", sign_only=False, alpha=0.05, proteins=False): #TODO: check if code is correct for n groups > 2
        '''
        This function calculates a t-test for selected proteins
        based on the current groups. Only executable with n group = 2.

        PARAMETERS
        ------------
        multiple_correction : str
            {bh (Benjamini/Hochberg), bon (bonferroni)} (default = bh)
        sign_only : boolean 
            only return significant values (default = False)
        alpha : float
            threshold for significance (default = 0.05)
        proteins : list 
            list of proteins to use (default = self.proteins)

        RETURNS
        ------------
        pvals : dictionary
            proteins as key and pvalue as value
        '''
        if type(proteins) != list:
            proteins = self.proteins

        spm = self.sample_protein_map(proteins=proteins)

        pvals = {}
        for p in proteins:
            anova_res = ss.f_oneway(*spm[p].values())
            pvals[p] = anova_res.pvalue

        to_pop = []
        for protein, p in pvals.items():
            if (not isinstance(p, numbers.Number)) or np.isnan(p) or p < 0 or p > 1:
                print("protein ", protein, " with value ", p, " was omitted")
                to_pop.append(protein)
        for i in to_pop:
            pvals.pop(i)
        
        if multiple_correction == "bh":
            fdc_pval = ss.false_discovery_control(list(pvals.values()))
            x=0
            for p in pvals:
                pvals[p] = fdc_pval[x]
                x+=1

        if multiple_correction == "bon":
            reject, pvalsbon, alphasidak, alphabonf = smsm.multipletests(list(pvals.values()),
                                        alpha=alpha,
                                        method="bonferroni")
            x=0
            for p in pvals:
                pvals[p] = pvalsbon[x]
                x+=1
        
        if sign_only:
            pvals = {p: v for p, v in pvals.items() if v <= alpha}

        return pvals

    def protein_batch_mann_whitney_u(self, proteins=False, sign_only=False, alpha=0.05, multiple_correction="bh"):
        '''
        This function conducts a mann whitney u test for selected proteins
        based on the current groups. Only executable with n group = 2.

        PARAMETERS
        ------------
        multiple_correction : str
            {bh (Benjamini/Hochberg), bon (bonferroni)} (default = bh)
        sign_only : boolean 
            only return significant values (default = False)
        alpha : float
            threshold for significance (default = 0.05)
        proteins : list 
            list of proteins to use (default = self.proteins)

        RETURNS
        ------------
        pvals : dictionary
            proteins as key and pvalue as value
        '''
        if type(proteins) != list:
            proteins = self.proteins

        spm = self.sample_protein_map(proteins=proteins)
        
        #testing !TWO GROUPS ONLY!
        pvals = {}
        for p in proteins:
            res = ss.mannwhitneyu(spm[p][self.groups[0]],
                         spm[p][self.groups[1]])
            pvals[p] = res.pvalue

        to_pop = []
        for protein, p in pvals.items():
            if (not isinstance(p, numbers.Number)) or np.isnan(p) or p < 0 or p > 1:
                print("protein ", protein, " with value ", p, " was omitted")
                to_pop.append(protein)
        for i in to_pop:
            pvals.pop(i)

        if multiple_correction == "bh":
            fdc_pval = ss.false_discovery_control(list(pvals.values()))
            x=0
            for p in pvals:
                pvals[p] = fdc_pval[x]
                x+=1

        if multiple_correction == "bon":
            reject, pvalsbon, alphasidak, alphabonf = smsm.multipletests(list(pvals.values()),
                                        alpha=alpha,
                                        method="bonferroni")
            x=0
            for p in pvals:
                pvals[p] = pvalsbon[x]
                x+=1

        if sign_only:
            pvals = {p: v for p, v in pvals.items() if v <= alpha}

        return pvals
    
    def linear_model(self, confounders=False, proteins=False, sign_only=False, alpha=0.05, multiple_correction="bh"):
        '''
        This function constructs a linear model for each protein
        based on the groups and choosen confounders.

        PARAMETERS
        ------------
        confounders : list 
            contains column names which are used as confounders
        multiple_correction : str
            {bh (Benjamini/Hochberg), bon (bonferroni)} (default = bh)
        sign_only : boolean 
            only return significant values (default = False)
        alpha : float
            threshold for significance (default = 0.05)
        proteins : list 
            list of proteins to use (default = self.proteins)

        RETURNS
        ------------
        pvals : dictionary
            proteins as key and pvalue as value
        '''
        if type(proteins) != list:
            proteins = self.proteins

        s = ""
        if type(confounders) == list:
            for n in range(len(confounders)):
                s += " + "+confounders[n]

        pvals = {}
        for p in proteins:
            model = smf.ols(p+"~"+self.group+s, data=self.df).fit()
            pvals[p] = model.pvalues[self.group+"[T.Control]"]

        to_pop = []
        for protein, p in pvals.items():
            if (not isinstance(p, numbers.Number)) or np.isnan(p) or p < 0 or p > 1:
                print("protein ", protein, " with value ", p, " was omitted")
                to_pop.append(protein)
        for i in to_pop:
            pvals.pop(i)

        if multiple_correction == "bh":
            fdc_pval = ss.false_discovery_control(list(pvals.values()))
            x=0
            for p in pvals:
                pvals[p] = fdc_pval[x]
                x+=1

        if multiple_correction == "bon":
            reject, pvalsbon, alphasidak, alphabonf = smsm.multipletests(list(pvals.values()),
                                        alpha=alpha,
                                        method="bonferroni")
            x=0
            for p in pvals:
                pvals[p] = pvalsbon[x]
                x+=1

        if sign_only:
            pvals = {p: v for p, v in pvals.items() if v <= alpha}

        return pvals

    def shapiro(self, multiple_correction="bh", proteins=False, sign_only=False, alpha=0.05, fdr=True):
        '''
        This function determines normality for selected proteins using the Shapiro-Wilk test.
        
        PARAMETERS
        ------------
        multiple_correction : str
            {bh (Benjamini/Hochberg), bon (bonferroni)} (default = bh)
        sign_only : boolean 
            only return significant values (default = False)
        alpha : float
            threshold for significance (default = 0.05)
        proteins : list 
            list of proteins to use (default = self.proteins)

        RETURNS
        ------------
        results : pandas dataframe 
            contains statistic, p-value, protein and group columns.
        '''
        if type(proteins) != list:
            proteins = self.proteins

        results = pd.DataFrame(columns=["statistic", "p-value", "protein", "group"])

        for p in proteins:
            group_1 = self.df[self.df[self.group] == self.groups[0]][p]
            group_2 = self.df[self.df[self.group] == self.groups[1]][p]

            group_1_stat, group_1_p = ss.shapiro(group_1)
            group_2_stat, group_2_p = ss.shapiro(group_2)

            results.loc[len(results)] = [group_1_stat, group_1_p, p, self.groups[0]]
            results.loc[len(results)] = [group_2_stat, group_2_p, p, self.groups[1]]

        pvals = results["p-value"].to_list()

        if multiple_correction == "bh":
            pvals = ss.false_discovery_control(pvals)
            
        if multiple_correction == "bon":
            reject, pvals, alphasidak, alphabonf = smsm.multipletests(pvals,
                                        alpha=alpha,
                                        method="bonferroni")

        results["p-value"] = pvals

        if sign_only:
            results = results[results["p-value"] <= alpha]

        return results

    def variance_test(self, multiple_correction="bh", method="barlett", proteins=False, sign_only=False, alpha=0.05, fdr=True):
        '''
        This function determines variance for selected proteins using a barlett or levene test.
        
        PARAMETERS
        ------------
        multiple_correction : str
            {bh (Benjamini/Hochberg), bon (bonferroni)} (default = bh)
        sign_only : boolean 
            only return significant values (default = False)
        alpha : float
            threshold for significance (default = 0.05)
        proteins : list 
            list of proteins to use (default = self.proteins)

        RETURNS
        ------------
        results : pandas dataframe
            contains statistic, p-value and protein columns.
        '''
        results = pd.DataFrame(columns=["statistic", "p-value", "protein"])

        if type(proteins) != list:
            proteins = self.proteins

        for p in proteins:
            if method == "levene":
                stat, pval = ss.levene(self.df[self.df[self.group] == self.groups[0]][p],
                        self.df[self.df[self.group] == self.groups[1]][p])
            elif method == "barlett":
                stat, pval = ss.bartlett(self.df[self.df[self.group] == self.groups[0]][p],
                        self.df[self.df[self.group] == self.groups[1]][p])
            else:
                print("Method not recognized: ", method)
                break
            results.loc[len(results)] = [stat, pval, p]

        pvals = results["p-value"].to_list()

        if multiple_correction == "bh":
            pvals = ss.false_discovery_control(pvals)

        if multiple_correction == "bon":
            reject, pvals, alphasidak, alphabonf = smsm.multipletests(pvals,
                                        alpha=alpha,
                                        method="bonferroni")

        results["p-value"] = pvals

        if sign_only:
            results = results[results["p-value"] <= alpha]

        return results
    
    #VISUALIZATION

    def qq(self, proteins=False):
        '''
        This function creates Q-Q plots for selected proteins.
        
        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        for p in proteins:
            for g in self.groups:
                fig = sm.qqplot(data=self.df[self.df[self.group] == g][p],
                        dist=ss.t,
                        line="45",
                        fit=True)
                plt.title(label="QQ plot for the "+p+" protein and the "+g+" group")
                plt.show()

    def ranked_quantification_plot(self, proteins=False):
        '''
        This function creates a ranked quantification plot for selected proteins.
        
        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        sums = [self.df[p].sum() for p in proteins]
        sums.sort()
        
        fig, ax = plt.subplots()
        ax.plot(range(len(sums)), sums)
        ax.set_title("Ranked quantification")
        ax.set_xlabel("proteins")
        ax.set_ylabel("total quantification")
        plt.show()

    def ranked_missing_plot(self, hline=50 ,proteins=False):
        '''
        This function creates a ranked missing plot for selected proteins.
        
        PARAMETERS
        ------------
        hline : int
            y value for placement of horizontal line (default = 50)
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        mis = self.missing()["n_missing"].tolist()
        mis.sort()
        
        fig, ax = plt.subplots()
        ax.plot(range(len(mis)), mis)
        ax.axhline(y=hline, color='red', linestyle='--')
        ax.set_title("missing values")
        ax.set_xlabel("proteins")
        ax.set_ylabel("number of samples with of missing value")
        plt.show()

    def missing_abundance_plot(self, method="total", proteins=False):
        '''
        This function creates a scatter plot of missing values and abundance.
        Includes a regression line as well.
        
        PARAMETERS
        ------------
        method : str
            {total, mean, median, min} (default = total)
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        mis = self.missing(proteins=proteins)["n_missing"].tolist()
        
        if method == "total": 
            sums = [self.df[p].sum() for p in proteins]
        if method == "mean": 
            sums = self.average_quant(na_zero=False)["avg_quant"].tolist() #TODO: Add proteins?
        if method == "median":
            sums = self.average_quant(na_zero=False, method=method)["avg_quant"].tolist()
        if method == "min":
            sums = [self.df[p].min() for p in proteins]

        fig, ax = plt.subplots()
        ax.scatter(mis, sums)
        
        slope, intercept, r, p, std_err = ss.linregress(mis, sums)
        lin_model = list(map(lambda xi: self.linear_calc(xi, slope, intercept), mis))
        plt.plot(mis, lin_model, color="k")

        ax.set_title("missing and abundance plot")
        ax.set_xlabel("number missing")
        ax.set_ylabel("abundance")
        plt.show()

    def mean_median_plot(self):#TODO: Add proteins
        '''
        This function plots the means and medians of protein intensity values
        '''
        mean = self.average_quant(na_zero=False)["avg_quant"].tolist()
        median = self.average_quant(na_zero=False, method="median")["avg_quant"].tolist()

        fig, ax = plt.subplots()
        ax.scatter(mean, median)
        
        slope, intercept, r, p, std_err = ss.linregress(mean, median)
        lin_model = list(map(lambda xi: self.linear_calc(xi, slope, intercept), mean))
        plt.plot(mean, lin_model, color="k")

        ax.set_title("mean median scatterplot")
        ax.set_xlabel("mean")
        ax.set_ylabel("median")
        plt.show()

    def obsmean_calcmean_plot(self, cutoff, threshold):#TODO: Add proteins
        '''
        This function plots the observed means and calculated means by MLE of protein intensity values

        PARAMETERS
        ------------
        cutoff : int
            cut-off of maximum likelihood
        threshold : int
            Threshold for the calculated mean
        '''
        obsmean = self.average_quant(na_zero=False)["avg_quant"].tolist()
        calcmean, calcsd = self.maximum_likelihood(cutoff)
        obsmean_threshold = []
        calcmean_threshold = []

        for i in range(len(calcmean)):
            if calcmean[i] < threshold:
                obsmean_threshold.append(obsmean[i])
                calcmean_threshold.append(calcmean[i])

        print(len(calcmean_threshold), " ", len(obsmean_threshold))

        fig, ax = plt.subplots()
        ax.scatter(obsmean_threshold, calcmean_threshold)

        ax.set_title("mean mean scatterplot")
        ax.set_xlabel("observed mean")
        ax.set_ylabel("calculated mean")
        plt.show()

    def read_count_boxplot(self, n_samples=False, proteins=False, log_transformed=True):
        '''
        This function creates boxplot for quantification 
        of selected proteins. 
        
        PARAMETERS
        ------------
        n_samples : List 
            contains two indices of samples (min, max) 
        log_transformed : Boolean
            indicates if data is log-transformed
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins
        if n_samples == False:
            n_samples = [0, len(self.df)]

        fig, ax = plt.subplots()

        data = [self.df[proteins].iloc[i].dropna().values for i in range(n_samples[0], n_samples[1])]

        bplot = ax.boxplot(data,
                    labels=self.df[self.id].tolist()[n_samples[0]:n_samples[1]],
                    patch_artist=True)
        ax.set_title("Boxplot of quantification")
        ax.set_xlabel("samples")
        plt.xticks(rotation=90)
        
        if log_transformed:
            ax.set_ylabel("log2 relative quantification")
        else:
            ax.set_ylabel("relative quantification")

        color_mapping = self.color_map()
        
        for patch, label in zip(bplot['boxes'], self.df[self.group].tolist()[n_samples[0]:n_samples[1]]):
            patch.set_facecolor(color_mapping[label])
            patch.set_edgecolor('black')
        
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in color_mapping.values()]
        ax.legend(handles, self.df[self.group].unique().tolist(), title="Category")
        plt.show()

    def density_plot(self, proteins=False):
        '''
        This function creates a density plot. 
        
        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        fig = sns.kdeplot(data=self.df[proteins].transpose(),
                    legend=False)
        if len(self.df) < 25:
            plt.legend(labels = self.df[self.id],
                    loc="upper right",
                        fontsize="5")
        plt.show(fig)

    def read_count_barplot(self,method="sum" ,proteins=False):
        '''
        This function creates barplot for quantification 
        of selected proteins. 
        
        PARAMETERS
        ------------
        methods : str
            method for read count {sum, median} (default = sum)
        proteins : list 
            list of proteins to use (default = self.proteins)

        DEPENDENCIES
        ------------
        function : color_map
        '''
        if type(proteins) != list:
            proteins = self.proteins

        if method == "sum":
            data = self.df[proteins].sum(axis=1)
        elif method == "median":
            data = self.df[proteins].median(axis=1)

        fig, ax = plt.subplots()

        color_mapping = self.color_map()
        colors = []
        for g in self.df[self.group]:
            colors.append(color_mapping[g])
        
        ax.bar(
            x=range(0,len(self.df[self.id])),
            height=data,
            tick_label=self.df[self.id],
            color=colors
        )

        ax.set_title("Barplot of total quantification")
        ax.set_xlabel("samples")
        ax.set_ylabel("relative quantification")
        plt.xticks(rotation=90)

        labels = list(color_mapping.keys())
        handles = [plt.Rectangle((0,0),1,1, color=color_mapping[label]) for label in labels]
        plt.legend(handles, labels)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')

        ax.tick_params(bottom=False, left=False)

        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#EEEEEE')
        ax.xaxis.grid(False)

        fig.tight_layout()

    def volcano_plot(self, control_group, treatment_group, lfc=False, pval=False, proteins=False, alpha=0.05):
        '''
        This function creates volcano plot using log-fold changes and pvalues 
        of selected proteins. 
        
        PARAMETERS
        ------------
        control_group : str
            contains group name of controls
        treatment_group : str
            contains group name of treatment
        lfc : list
            contains fold change values
        pval : list
            contains p-values
        alpha : float
            threshold for significance (default = 0.05)
        proteins : list 
            list of proteins to use (default = self.proteins)

        DEPENDENCIES
        ------------
        function : protein_batch_ttest
        function : fold_change
        '''
        if type(proteins) != list:
            proteins = self.proteins
        if type(lfc) != list:
            lfc = self.fold_change(proteins=proteins,
                                   log=True,
                                   control_group=control_group,
                                   treatment_group=treatment_group)
            lfc = list(lfc.values())
        if type(pval) != list:
            pval = self.protein_batch_mann_whitney_u(proteins=proteins) #TODO: make multiple tests available
            pval = list(pval.values())

        fig, ax = plt.subplots()

        colors = []
        x=0
        for l in lfc:
            if l >= self.log_fold_thres and pval[x] >= alpha or l <= -self.log_fold_thres and pval[x] >= alpha:
                colors.append("green")
            elif l < self.log_fold_thres and l > -self.log_fold_thres and pval[x] < alpha:
                colors.append("blue")
            elif l >= self.log_fold_thres and pval[x] < alpha or l <= -self.log_fold_thres and pval[x] < alpha:
                colors.append("red")
                ax.annotate(proteins[x], (l, -np.log10(pval[x])))
            else:
                colors.append("grey")
            x+=1

        pval = -np.log10(pval)

        ax.scatter(lfc, pval, alpha=0.5, c=colors)
        ax.axhline(y=-np.log10(alpha), color='red', linestyle='--')
        ax.axvline(x=self.log_fold_thres, color='gray', linestyle='--')
        ax.axvline(x=-self.log_fold_thres, color='gray', linestyle='--')
        ax.set_xlabel('Log Fold Change')
        ax.set_ylabel('-log10(p-value)')
        ax.set_title('Volcano Plot')
        plt.show()

    def distance_heatmap(self, method="euclidean", control_group=False, treatment_group=False, proteins=False):
        '''
        This function creates heatmap based on a distance matrix.
        
        PARAMETERS
        ------------
        method : str
            distance metric to use {euclidean, manhattan, poisson} (default = euclidean)
        proteins : list 
            list of proteins to use (default = self.proteins)

        DEPENDENCIES
        ------------
        function : euclidian_distance_matrix
        function : manhattan_distance_matrix
        function : poisson_distance_matrix
        '''
        if type(proteins) != list:
            proteins = self.proteins
        if method == "euclidean":
            dm = self.euclidian_distance_matrix(proteins=proteins)
        elif method == "manhattan":
            dm = self.manhattan_distance_matrix(proteins=proteins)
        elif method == "poisson":
            dm = self.poisson_distance_matrix(proteins=proteins)

        row_linkage = linkage(dm, method='average')
        if control_group != False and treatment_group != False:
            class_colors = {
                control_group: '#4daf4a',  # green
                treatment_group: '#e41a1c'   # red
            }
            col_colors = [class_colors[label] for label in list(self.df[self.group])]

            g = sns.clustermap(dm, 
                        row_linkage=row_linkage,
                        annot=False,
                        col_colors=col_colors,
                        xticklabels=self.df[self.id],
                        yticklabels=self.df[self.id])
            
            legend_patches = [mpatches.Patch(color=color, label=label) for label, color in class_colors.items()]

            g.fig.legend(
                handles=legend_patches,
                title='Sample Class',
                loc='upper right',
                bbox_to_anchor=(1.15, 0.9)
            )   
        else:
            sns.clustermap(dm, 
                        row_linkage=row_linkage,
                        annot=False,
                        xticklabels=self.df[self.id],
                        yticklabels=self.df[self.id])

    def multi_dimensional_scaling_plot(self, distance_matrix): 
        '''
        This function creates a multi dimensional scaling plot
        based on a distance matrix. Supports up to two groups.
        
        PARAMETERS
        ------------
        distance_matrix : matrix

        DEPENDENCIES
        ------------
        function : color_map
        '''
        scaling = self.multi_dimensional_scaling(distance_matrix)

        color_mapping = self.color_map()
        colors = []
        for g in self.df[self.group]:
            colors.append(color_mapping[g])
        
        x = [i[0] for i in scaling]
        y = [i[1] for i in scaling]

        fig, ax = plt.subplots()
        ax.scatter(x,y,c=colors)

        labels = list(color_mapping.keys())
        handles = [plt.Rectangle((0,0),1,1, color=color_mapping[label]) for label in labels]
        plt.legend(handles, labels)

        ax.set_title('multi dimensional scaling plot')
        plt.show()

    def pca_plot(self, proteins=False):
        '''
        This function creates a scatter plot of the first
        two pricipal components.
        
        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)

        DEPENDENCIES
        ------------
        function : pca
        '''
        if type(proteins) != list:
            proteins = self.proteins

        pca_obj, pca_vals = self.pca(proteins=proteins)

        color_mapping = self.color_map()
        colors = []
        for g in self.df[self.group]:
            colors.append(color_mapping[g])
        
        x = [i[0] for i in pca_vals]
        y = [i[1] for i in pca_vals]

        fig, ax = plt.subplots()
        ax.scatter(x,y,c=colors)

        labels = list(color_mapping.keys())
        handles = [plt.Rectangle((0,0),1,1, color=color_mapping[label]) for label in labels]
        plt.legend(handles, labels)

        ax.set_xlabel('1st principal component')
        ax.set_ylabel('2nd principal component')
        ax.set_title('pca plot')
        plt.show()

    def tsne_plot(self, distance_matrix, perplexity=30):
        '''
        This function creates a scatter plot of the t-SNE function.
        
        PARAMETERS
        ------------
        distance_matrix : matrix
        perplexity : int 
            must be less than n samples (default = 30)

        DEPENDENCIES
        ------------
        function : tsne
        '''
        tsne_res = self.tsne(distance_matrix=distance_matrix,
                             perplexity = perplexity)

        color_mapping = self.color_map()
        colors = []
        for g in self.df[self.group]:
            colors.append(color_mapping[g])
        
        x = [i[0] for i in tsne_res]
        y = [i[1] for i in tsne_res]

        fig, ax = plt.subplots()
        ax.scatter(x,y,c=colors)

        labels = list(color_mapping.keys())
        handles = [plt.Rectangle((0,0),1,1, color=color_mapping[label]) for label in labels]
        plt.legend(handles, labels)

        ax.set_title('t-SNE plot')
        plt.show()

    def umap_plot(self, proteins=False):
        '''
        This function creates a scatter plot of the 2d UMAP function.
        
        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)

        DEPENDENCIES
        ------------
        function : umap_2d
        '''
        if type(proteins) != list:
            proteins = self.proteins

        umap_res = self.umap_2d(proteins=proteins)

        color_mapping = self.color_map()
        colors = []
        for g in self.df[self.group]:
            colors.append(color_mapping[g])
        
        x = [i[0] for i in umap_res]
        y = [i[1] for i in umap_res]

        fig, ax = plt.subplots()
        ax.scatter(x,y,c=colors)

        labels = list(color_mapping.keys())
        handles = [plt.Rectangle((0,0),1,1, color=color_mapping[label]) for label in labels]
        plt.legend(handles, labels)

        ax.set_title('umap plot')
        plt.show()

    def protein_correlation_heatmap(self, proteins=False):
        '''
        This function creates correlation heatmap based on protein data.

        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        corr = self.df[proteins].corr()
        sns.clustermap(corr)

    def sample_correlation_heatmap(self, control_group=False, treatment_group=False, proteins=False):
        '''
        This function creates correlation heatmap on protein data of samples.

        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)

        DEPENDENCIES
        ------------
        function : peason_correlation
        '''
        if type(proteins) != list:
            proteins = self.proteins

        pcors, ppvals = self.pearson_correlation(proteins=proteins)

        row_linkage = linkage(pcors, method='average')

        if control_group != False and treatment_group != False:
            class_colors = {
                control_group: '#4daf4a',  # green
                treatment_group: '#e41a1c'   # red
            }
            col_colors = [class_colors[label] for label in list(self.df[self.group])]

            g = sns.clustermap(pcors, 
                        row_linkage=row_linkage,
                        annot=False,
                        col_colors=col_colors,
                        xticklabels=self.df[self.id],
                        yticklabels=self.df[self.id])
            
            legend_patches = [mpatches.Patch(color=color, label=label) for label, color in class_colors.items()]

            g.fig.legend(
                handles=legend_patches,
                title='Sample Class',
                loc='upper right',
                bbox_to_anchor=(1.15, 0.9)
            )   
        else:
            sns.clustermap(pcors,
                        xticklabels=self.df[self.id],
                        yticklabels=self.df[self.id])

    #HELPER FUNCTIONS

    def color_map(self):
        '''
        This function creates a dictionary with colors 
        alternating between two colors based on the groups.

        RETURNS
        ------------
        color_mapping : dictionary 
            group as key and color as value
        '''
        color_mapping = {}
        x=True
        for g in self.groups:
            if x:
                color_mapping[g] = "#1f77b4"
                x=False
            else:
                color_mapping[g] = "#ff7f0e"
                x=True
        return color_mapping

    def sample_protein_map(self, proteins=False):
        '''
        This function maps proteins based on the proteins and samples.
        
        PARAMETERS
        ------------
        proteins : list 
            list of proteins to use (default = self.proteins)

        RETURNS
        ------------
        nested dictionary : dictionary 
            proteins as keys and dictionaries with groups and lists of protein quantifications
        '''
        if type(proteins) != list:
            proteins = self.proteins

        #Data prep
        t = {}
        for p in proteins:
            t[p] = {} #Make a new instance for each protein
            x=0
            for g in self.df[self.group]:
                if not pd.isna(self.df[p].iloc[x]):
                    if g in t[p]: #Add data based on group
                        t[p][g].append(self.df[p].iloc[x])
                    else:
                        t[p][g] = [self.df[p].iloc[x]]
                x+=1
        return t
    
    def linear_calc(self, x, slope, intercept):
        '''
        This function calculates a prediction using a linear model

        PARAMETERS
        ------------
        x : numeric 
            x value
        slope : numeric 
            value of the slope of the linear model
        intercept :  numeric 
            value of the intercept of the linear model

        RETURNS
        ------------
        y : numeric 
            predicted value
        '''
        return slope * x + intercept
    
    def neg_log_likelihood(self, params, cutoff, obs_vals):
        '''
        This function computes the negative log-likelihood for a censored normal distribution model 
        given observed data, a cutoff value, and distribution parameters.

        Parameters:
        ----------
        params : list or tuple of float
            A sequence containing two values: the mean and standard deviation of the 
            normal distribution [mean, sd].
        cutoff : float
            The censoring threshold. Values below this are considered censored in the 
            likelihood calculation.
        obs_vals : list of float
            The list of observed (non-censored) values.

        Returns:
        -------
        float
            The negative log-likelihood value. Returns infinity if the standard deviation 
            is not positive.
        '''
        mean, sd = params
        if sd <= 0:
            return np.inf

        else:
            z = (cutoff - mean) / sd
            lprob_censored = np.log(ss.norm.cdf(z))
            lobs_censored = np.sum(ss.norm.logpdf(obs_vals, loc=mean, scale=sd))
            return -(len(obs_vals) * lprob_censored * lobs_censored)
        
    #PIPELINES

    def pipeline(self, imp_method, norm_method, threshold, log_transform,
                control_group, treatment_group, 
                log_transformend=False, volcanoplot=False ,test="mw",
                laplace_value=1, nn_knn=4, order="lin", n_iter=1, multiple_correction="bh",
                confounders=[], mrf_n_iter=2, modeling=False, modeltype="LR"):
        '''
        This function runs an entire pipeline for DAP identification

        Parameters:
        ----------
        imp_method : str
            imputation method {knn, avg, laplace, mrf}
        norm_method : str
            normalization method {median, cloess, quantile, ti, vsn, linear, ma_linear, mrn, skl}
        threshold : int
            cut-off of missing value percentage 
        log_transform : boolean
            indicates if data needs to be log transformend
        control_group : str
            contains group name of controls
        treatment_group : str
            contains group name of treatment
        log_transformend : boolean
            indicates if data is already log transformend (default = False)
        volcanoplot : boolean
            indicates if a volcano plot should be constructed (default = False)
        test : str
            statistical test to be used {mw, lin, tt, anova} (default = mw)
        laplace_value : numeric
            value to use in laplace imputation (default = 1)
        order : str
            order in which (l)og-transformation, (i)mputation and (n)ormalization should be done,
            any combination of [i, l, n] (default = lin)
        n_iter : int 
            indicates the number of iterations relevant imputation method (default = 1)
        multiple_correction : str 
            multiple correction method to use {bh, bonn, ""} (default = bh)
        confounders : list 
            column names to use as confounders in linear model (default = [])
        mrf_n_iters : int
            number of iterations of the miss random forest imputation technique (default = 2)
        nn_knn : int
            number of neighbours for the knn imputation method (default = 4)
        modeling : boolean
            indicates if a LR model should be constructed (uses LOOCV) and evaluated (default = False)
            
        Returns:
        -------
        dap : dictionary 
            DAP's names as key and list with p-value and log-fold change as value
        string : str
            representation of methods

        Notes:
        -------
        - supplied log transformend data will always be exponentiated 
        - current object should hold correct list of proteins
        - log trans-formation is ignored if set to False in order
        - imputation comes before DAP identification and modeling, use with care!
        '''
        arguments = locals()

        #Filter missing values
        self.missing_filtering(threshold=threshold, percentage=True)
        
        #Exponentiate log transformend data
        if log_transformend:
            self.exponentiate()
        
        #Do the imputation, normalization and log-transformation
        for i in order:
            if i == "i":
                self.pipeline_imputation(imp_method=imp_method,
                                         nn_knn=nn_knn,
                                         laplace_value=laplace_value,
                                         mrf_n_iter=mrf_n_iter)
            if i == "l":
                self.pipeline_log_transform(log_transform=log_transform)
            if i == "n":
                self.pipeline_normalization(norm_method=norm_method,
                                            n_iter=n_iter)

        #Calculate p-values
        pvals_dict = self.pipeline_test(test=test, 
                                        multiple_correction=multiple_correction,
                                        confounders=confounders)
        pvals = list(pvals_dict.values())
        ps = list(pvals_dict.keys())

        #Calculate log fold changes
        if log_transform:
            logs = self.fold_change(treatment_group=treatment_group, 
                                    control_group=control_group, 
                                    logged=True, log=True)
        else:
            logs = self.fold_change(treatment_group=treatment_group,
                                    control_group=control_group, 
                                    logged=False, log=True) 
        logs = [logs[p] for p in ps]

        #Construct volcano plot
        if volcanoplot:
            self.volcano_plot(treatment_group=treatment_group, 
                           control_group=control_group, 
                           proteins=ps, lfc=logs, pval=pvals)
        
        #Identify the DAP's
        daps = []
        daps_res = {}
        for i in range(len(pvals)):
            if pvals[i] <= 0.05 and logs[i] >=self.log_fold_thres:
                daps.append(ps[i])
                daps_res[ps[i]] = [pvals[i], logs[i]]
            if pvals[i] <= 0.05 and logs[i] <=-self.log_fold_thres:
                daps.append(ps[i])
                daps_res[ps[i]] = [pvals[i], logs[i]]
        
        methods_str = "_".join([str(arguments[arg]) for arg in arguments])

        if modeling:
            if len(daps) > 0:
                self.multiple_protein_aucroc(control_group=control_group,
                                            treatment_group=treatment_group,
                                            pca=False,
                                            proteins=daps,
                                            modeltype=modeltype
                                            )
            else:
                print("No DAP's found, no modeling possible")

        return daps_res, methods_str

    def pipeline_imputation(self, imp_method, nn_knn, laplace_value, mrf_n_iter):
        '''
        Helper function for the pipeline method, executes the imputation method

        Parameters:
        ----------
        imp_method : str
            imputation method {knn, avg, laplace, mrf}
        laplace_value : numeric
            value to use in laplace imputation (default = 1)
        nn_knn : int
            number of neighbours for the knn imputation method
        mrf_n_iter : int
            number of iterations of the miss random forest imputation technique
        '''
        if imp_method == "knn":
            self.knn_imputation(n_neighbors=nn_knn)
        if imp_method == "avg":
            self.avg_imputation()
        if imp_method == "laplace":
            self.laplace(value=laplace_value)
        if imp_method == "mrf":
            self.miss_forest_imputation(n_iter=mrf_n_iter)

    def pipeline_normalization(self, norm_method, n_iter):
        '''
        Helper function for the pipeline method, executes the normalization method

        Parameters:
        ----------
        norm_method : str
            normalization method {median, cloess, quantile, ti, vsn, linear, ma_linear, mrn, skl}
        n_iter : int 
            indicates the number of iterations relevant imputation method
        '''
        if norm_method == "median":
            self.median_normalize()
        if norm_method == "cloess":
            self.cyclic_loess(n_iter=n_iter)
        if norm_method == "quantile":
            self.quantile_normalization()
        if norm_method == "ti":
            self.total_intensity_normalization()
        if norm_method == "vsn":
            self.variance_stabilizing_normalization(tol=0.01, max_iter=n_iter)
        if norm_method == "linear":
            self.ma_linear_regression_normalization(n_iter=n_iter)
        if norm_method == "malinear":
            self.ma_linear_regression_normalization(n_iter=n_iter)
        if norm_method == "mrn":
            self.median_ratio_normalization()
        if norm_method == "skl":
            self.skl_normalize()

    def pipeline_log_transform(self, log_transform):
        '''
        Helper function for the pipeline method, executes the log-transformation method

        Parameters:
        ----------
        log_transform : boolean
            indicates if data needs to be log transformend
        '''
        if log_transform:
            m = self.df[self.proteins].min().min()
            if m <= 0: #Make all values positive and add small laplace value
                self.df[self.proteins] = self.df[self.proteins] + abs(m) + +1e-6
                print("ALTERED DATA BEFORE LOG-TRANSFORMATION")
                print("min value: ", m) 
            self.log_transform()

    def pipeline_test(self, test, multiple_correction, confounders):
        '''
        Helper function for the pipeline method, executes the statistical test

        Parameters:
        ----------
        test : str
            statistical test to be used {mw, lin, tt, anova}
        multiple_correction : str
            multiple correction method to use {bh, bonn, ""}
        confounders : list
            column names to use as confounders in linear model (default = [])
        '''
        if test == "mw":
            return self.protein_batch_mann_whitney_u(multiple_correction=multiple_correction)
        if test == "lin":
            return self.linear_model(confounders=confounders,
                                     multiple_correction=multiple_correction)
        if test == "tt":
            return self.protein_batch_ttest(multiple_correction=multiple_correction)
        if test == "anova":
            return self.protein_batch_anova(multiple_correction=multiple_correction)
        
    def grid_pipeline(self, imp_methods, norm_methods, thresholds, 
                control_group, treatment_group, log_transform = [True],
                log_transformend=False, volcanoplot=False ,test="mw",
                laplace_values=[1], nn_knn=[4], orders=["lin"], n_iter=1, multiple_correction="bh",
                confounders=[], mrf_n_iters=[2], modeling=False):
        '''
        This function runs an grid for specified settings for the entire pipeline for DAP identification

        Parameters:
        ----------
        imp_methods : list
            imputation methods {knn, avg, laplace, mrf}
        norm_methods : list
            normalization methods {median, cloess, quantile, ti, vsn, linear, ma_linear, mrn, skl}
        thresholds : list
            cut-offs of missing value percentage 
        log_transform : list
            booleans indicating if data needs to be log transformend (default = [True])
        control_group : str
            contains group name of controls
        treatment_group : str
            contains group name of treatment
        log_transformend : boolean
            indicates if data is already log transformend (default = False)
        volcanoplot : boolean
            indicates if a volcano plot should be constructed (default = False)
        test : str
            statistical test to be used {mw, lin, tt, anova} (default = mw)
        laplace_value : numeric
            value to use in laplace estimator (default = 1)
        order : str
                order in which (l)og-transformation, (i)mputation and (n)ormalization should be done,
                any combination of [i, l, n] (default = lin)
        n_iter : int
            indicates the number of iterations relevant imputation method (default = 1)
        multiple_correction : str
            multiple correction method to use {bh, bonn, ""} (default = bh)
        confounders : list
            column names to use as confounders in linear model (default = [])
        mrf_n_iters : list
            number of iterations of the miss random forest imputation technique (default = [2])
        nn_knn : list
            number of neighbours for the knn imputation method (default = [4])
        
        Returns:
        -------
        results : dataframe
            contains boolean values indicating if proteins are identified as DAP and columns for settings
        '''
        
        methods = ["order","norm_method", "threshold", "logtrans", "imp_method"]
        results = pd.DataFrame(columns=self.proteins+methods)

        for order in orders:
            for norm_method in norm_methods:
                for threshold in thresholds:
                    for logtrans in log_transform:
                        for imp_method in imp_methods:
                            if imp_method == "knn":
                                for n in nn_knn:
                                    tp = self.get_proteins().copy()
                                    tdf = self.get_data().copy()
                                    dic, st = self.pipeline(imp_method="knn", norm_method=norm_method, threshold=threshold, log_transform=logtrans,
                                                        control_group=control_group, treatment_group=treatment_group, 
                                                        log_transformend=log_transformend, volcanoplot=volcanoplot ,test=test,
                                                        nn_knn=n, order=order, n_iter=n_iter, multiple_correction=multiple_correction,
                                                        confounders=confounders, modeling=modeling)
                                    row_index = len(results)
                                    row_data = {p: 1 if p in dic.keys() else 0 for p in self.proteins}
                                    row_data.update(dict(zip(methods, [str(order), str(norm_method), str(threshold), str(logtrans), str(imp_method)])))

                                    results.loc[row_index] = row_data

                                    print(st)
                                    self.set_data(data=tdf)
                                    self.set_proteins(tp)

                            elif imp_method == "laplace":
                                for n in laplace_values:
                                    tp = self.get_proteins().copy()
                                    tdf = self.get_data().copy()
                                    dic, st = self.pipeline(imp_method="laplace", norm_method=norm_method, threshold=threshold, log_transform=logtrans,
                                                        control_group=control_group, treatment_group=treatment_group, 
                                                        log_transformend=log_transformend, volcanoplot=volcanoplot ,test=test,
                                                        laplace_value=n, order=order, n_iter=n_iter, multiple_correction=multiple_correction,
                                                        confounders=confounders, modeling=modeling)
                                    row_index = len(results)
                                    row_data = {p: 1 if p in dic.keys() else 0 for p in self.proteins}
                                    row_data.update(dict(zip(methods, [str(order), str(norm_method), str(threshold), str(logtrans), str(imp_method)])))

                                    results.loc[row_index] = row_data

                                    print(st)
                                    self.set_data(data=tdf)
                                    self.set_proteins(tp)

                            elif imp_method == "mrf":
                                for n in mrf_n_iters:
                                    tp = self.get_proteins().copy()
                                    tdf = self.get_data().copy()
                                    dic, st = self.pipeline(imp_method="mrf", norm_method=norm_method, threshold=threshold, log_transform=logtrans,
                                                        control_group=control_group, treatment_group=treatment_group, 
                                                        log_transformend=log_transformend, volcanoplot=volcanoplot ,test=test,
                                                        mrf_n_iter=n, order=order, n_iter=n_iter, multiple_correction=multiple_correction,
                                                        confounders=confounders, modeling=modeling)
                                    row_index = len(results)
                                    row_data = {p: 1 if p in dic.keys() else 0 for p in self.proteins}
                                    row_data.update(dict(zip(methods, [str(order), str(norm_method), str(threshold), str(logtrans), str(imp_method)])))

                                    results.loc[row_index] = row_data

                                    print(st)
                                    self.set_data(data=tdf)
                                    self.set_proteins(tp)
        
                            else:
                                tp = self.get_proteins().copy()
                                tdf = self.get_data().copy()
                                dic, st = self.pipeline(imp_method=imp_method, norm_method=norm_method, threshold=threshold, log_transform=logtrans,
                                                        control_group=control_group, treatment_group=treatment_group, 
                                                        log_transformend=log_transformend, volcanoplot=volcanoplot ,test=test,
                                                        order=order, n_iter=n_iter, multiple_correction=multiple_correction,
                                                        confounders=confounders, modeling=modeling)
                            
                                row_index = len(results)
                                row_data = {p: 1 if p in dic.keys() else 0 for p in self.proteins}
                                row_data.update(dict(zip(methods, [str(order), str(norm_method), str(threshold), str(logtrans), str(imp_method)])))

                                results.loc[row_index] = row_data

                                print(st)
                                self.set_data(data=tdf)
                                self.set_proteins(tp)
        return results

    #MACHINE LEARNING FUNCTIONS

    def loocv(self, modeltype="LR", pca=False, smote=False, proteins=False):
        '''
        This function creates a machine-learning model and evaluates it using LOOCV

        Parameters:
        ----------
        modeltype : str
            machine learning model to use {LR, RF, NB, SVM} (default = LR)
        pca : boolean
            indicates if data needs to be PCA transformend (default = false)
        smote : boolean
            indicates if synthetic data upsampling using the SMOTE technique 
            should be performend (default = False)
        proteins : list 
            list of proteins to use (default = self.proteins)

        Notes
        ----------
        - PCA transformation uses the first 2 dimensions
        '''
        if type(proteins) != list:
            proteins = self.proteins

        y_true, y_pred = list(), list()
        cv = LeaveOneOut()

        for train_ix, test_ix in cv.split(self.df):
            # split data
            X_train, X_test = self.df.loc[train_ix.tolist(), proteins], self.df.loc[test_ix.tolist(), proteins]
            y_train, y_test = self.df.loc[train_ix.tolist(), self.group], self.df.loc[test_ix.tolist(), self.group]

            if pca:
                pca_res = PCA(n_components=2)
                pca_res.fit(X_train)
                X_train = pca_res.transform(X_train)
                X_test = pca_res.transform(X_test)

            if smote:
                sm = SMOTE(random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)

            # fit model
            if modeltype == "LR":
                model = self.logistic_regression(X_train, y_train)
            if modeltype == "RF":
                model = self.random_forest(X_train, y_train)
            if modeltype == "NB":
                model = self.naive_bayes(X_train, y_train)
            if modeltype =="SVM":
                model = self.svm_linear(X_train, y_train)
            
            # make predictions
            predictions, predictions_prob = self.model_predict(x_test=X_test, model=model)
            # store
            y_true.append(y_test[test_ix[0]])
            y_pred.append(predictions[0])
        return y_true, y_pred
    
    def logistic_regression(self, x_train, y_train, multiclass=False):
        '''This function creates a logistic regression model

        Parameters
        ----------
        x_train : pandas dataframe object
            Data on which the model should be trained
        y_train : array
            List containing class labels
        multiclass : int
            Binary indicating whether it involves a multiclass problem

        Returns
        -------
        logisticRegr : sklearn LogisticRegression object
            Logistic resgression model
        '''

        if multiclass:
            logisticRegr = LogisticRegression(multi_class='multinomial', max_iter=1000000000)
        else:
            logisticRegr = LogisticRegression(max_iter=1000000000)
        logisticRegr.fit(x_train, y_train)
        return logisticRegr
    
    def naive_bayes(self, x_train, y_train):
        '''This function creates a naíve bayes model

        Parameters
        ----------
        x_train : pandas dataframe object
            Data on which the model should be trained
        y_train : array
            List containing class labels
        
        Returns
        -------
        model : sklearn Guassian naïve bayes object
        '''

        model = GaussianNB()
        model.fit(x_train, y_train)
        return model
    
    def random_forest(self, x_train, y_train):
        '''This function creates a random forest model

        Parameters
        ----------
        x_train : pandas dataframe object
            Data on which the model should be trained
        y_train : array
            List containing class labels
        
        Returns
        -------
        model : sklearn random forest object
        '''

        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        return model
    
    def svm_linear(self, x_train, y_train):
        #model = svm.SVC(kernel="linear", probability=True)
        model = svm.SVC(probability=True)
        model.fit(x_train, y_train)
        return model
    
    def model_predict(self, model, x_test, multiclass=False):
        '''This function handles predictions using a given model

        Parameters
        ----------
        model : sklearn model object
        x_test : pandas dataframe object
            Data with whom the model should be tested
        multiclass : int
            Binary indicating whether it involves a multiclass problem

        Returns
        -------
        Predictions : array
            List with class predictions
        Predictions_prob : array
            List with probabilities of the predictions
        '''

        if multiclass:
            predictions = model.predict(x_test) #predict_proba
            predictions_prob = model.predict_proba(x_test)
            return predictions, predictions_prob
        else:
            predictions = model.predict(x_test)
            predictions_prob = model.predict_proba(x_test)
            return predictions, predictions_prob
        
    def auc(self, y_test, predictions):
        '''This function calculates the AUC of the ROC

        Parameters
        ----------
        y_test : list
            Contains testing class labels
        Predictions : list
            List with class predictions

        Returns
        -------
        auc score : float
        '''

        return metrics.roc_auc_score(y_test, predictions)
    
    def multiple_single_protein_aucroc(self, treatment_group, control_group, modeltype="LR", proteins=False, smote=False):
        '''
        This function makes a classifier for each individual protein

        Parameters
        ----------
        control_group : str
            contains group name of controls
        treatment_group : str
            contains group name of treatment
        modeltype : str
            machine learning model to use {LR, RF, NB, SVM} (default = LR)
        pca : boolean
            indicates if data needs to be PCA transformend (default = false)
        smote : boolean
            indicates if synthetic data upsampling using the SMOTE technique 
            should be performend (default = False)
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        mapping = {treatment_group : 0, control_group : 1}

        for p in proteins:
            y_true, y_pred = self.loocv(proteins=[p], modeltype=modeltype, smote=smote)
            
            y_true = [mapping[i] for i in y_true]
            y_pred = [mapping[i] for i in y_pred]

            fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
            auc = round(metrics.roc_auc_score(y_true, y_pred), 4)
            print(p, " : ", auc)

    def multiple_protein_aucroc(self, treatment_group, control_group, modeltype="LR", pca=False, proteins=False, smote=False):
        '''
        Contructs a classifier based on selected proteins

        Parameters
        ----------
        control_group : str
            contains group name of controls
        treatment_group : str
            contains group name of treatment
        modeltype : str
            machine learning model to use {LR, RF, NB, SVM} (default = LR)
        pca : boolean
            indicates if data needs to be PCA transformend (default = false)
        smote : boolean
            indicates if synthetic data upsampling using the SMOTE technique 
            should be performend (default = False)
        proteins : list 
            list of proteins to use (default = self.proteins)
        '''
        if type(proteins) != list:
            proteins = self.proteins

        mapping = {treatment_group : 1, control_group : 0}

        y_true, y_pred = self.loocv(proteins=proteins, modeltype=modeltype, pca=pca, smote=smote)
        
        y_true = [mapping[i] for i in y_true]
        y_pred = [mapping[i] for i in y_pred]

        auc = round(metrics.roc_auc_score(y_true, y_pred), 4)
        print("AUC : ", auc)
        print(classification_report(y_true, y_pred, target_names=[treatment_group, control_group]))