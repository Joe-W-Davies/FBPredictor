import pandas as pd
import numpy as np
import xgboost as xg
from scipy.stats import binom
import shap

class BorutaShap(object):
    '''
    Use the Boruta feature selection with shapley values to systematically drop features
   
    attributes
    ----------
    :param X_train: DataFrame containing all X features required for initial training
    :param y_train: target columns (electron or PU flag)
    :param w_train: weights for training
    :param all_vars: list of all variables being considered for droppping
    :param i_ters: max number of times to run slimming algorithm. Useful if eventually the feature importance
                   for the every real variable is always higher than the shadow variable (avoid inf loop)
    :param n_trainings: number of times run classifier training and calculate importances
    :param max_vars_removed: max number of variables that may be removed
    :param running_vars: list of all variables that are to be kept. Updated as algo runs
    :param running_shadow_vars: list of current shadow variables. Updated as algo runs, based on running_vars
    :param tolerance: fraction of each column to be randomly shuffled when creating shadow variables. 
                      High tolerance mean less conservative algo
    :param p_remove: lower probability threshold to remove feature. Default is set to (mean -  2 * of the binomial distribution)
                     obtained from variable "hits", where hit == var importance was better than highest shadow feature importance
    :param keep_shadows: dont throw away shadow features when original features is removed. Makes algo less conservative
    '''

    def __init__(
            self, 
            X_train: pd.DataFrame | np.ndarray, 
            y_train: pd.DataFrame | np.ndarray, 
            all_vars: list[str], 
            w_train: pd.DataFrame | np.ndarray, 
            i_iters: int=5, 
            n_trainings: int=50, 
            max_vars_removed: int=10, 
            tolerance: int=0.6, 
            train_params: dict={}, 
            keep_shadows: bool=True
        ):

        self.X_train             = pd.DataFrame(X_train, columns=all_vars)
        self.y_train             = y_train
        self.w_train             = w_train
        self.all_vars            = all_vars
        self.i_iters             = i_iters
        self.n_trainings         = n_trainings
        self.max_vars_removed    = max_vars_removed
        self.running_vars        = all_vars
        self.running_shadow_vars = ['shadow_'+str(var) for var in all_vars] 
        self.tolerance           = tolerance
        self.keep_shadows        = keep_shadows
        self.train_params        = train_params

        self.p_hit = 0.5
        binom_mean = n_trainings * self.p_hit
        binom_std  = (n_trainings * (1-self.p_hit) * self.p_hit)**0.5
        #print (binom_mean - (1*binom_std))
        hit_threshold       = (binom_mean - (1*binom_std))
        if hit_threshold < 0: raise ValueError('hit threshold was below zero! either increase n_trainings, or change threshold e.g. 1 std -> 2 std')
        self.hit_threshold       = round(hit_threshold)
        print ('lowest N_hits for removing a feature is: ', self.hit_threshold)

        #assert each element in all_vars is in X_train.columns


    def update_vars(self, varrs: list[str]) -> None:
        '''
        Update current variables to train on, for both real and shadow sets
        '''
        self.running_vars = varrs
        if self.keep_shadows: self.running_shadow_vars = ['shadow_'+str(var) for var in self.all_vars]
        else: self.running_shadow_vars = ['shadow_'+str(var) for var in varrs]


    def create_shadow(self) -> pd.DataFrame:
        '''
        Take all X variables, creating copies and randomly shuffling them (or a subset of them)

        :return: dataframe 2x width and the names of the shadows for removing later
        '''

        real_running_x = self.X_train[self.running_vars]
        final_shuf_c  = []
        split_id = int(self.tolerance*real_running_x.shape[0])
        if self.keep_shadows: s_vars = self.all_vars
        else: s_vars = real_running_x.columns
        for c in s_vars:
            shuf_c, nominal_c = np.split(self.X_train[c], [split_id])
            np.random.shuffle(np.array(shuf_c))
            final_shuf_c.append(pd.concat([nominal_c, shuf_c]))

        #concat partially mixed columns into a single df
        x_shadow = pd.concat(final_shuf_c, axis=1)#, names = self.running_shadow_vars)
        x_shadow.columns = self.running_shadow_vars
        x_shadow = x_shadow.sample(frac=1) #disperse shuffled rows in shadow df

        final_df = pd.concat([real_running_x, x_shadow], axis=1)
        #print ('final df columns: {}'.format(list(final_df.columns)[:]))

            
        return pd.concat([real_running_x, x_shadow], axis=1)


    def run_trainings(self) -> int:
        '''
        Run classifier training for n_iters. Return the importances for each feature averaged over n_iters.
        We run this multiple times since the feature importance ranking is not deterministic.
        Note we need to set up the dict first with all real + shadow features still being considered,
        since the faetures importances for some features are zero and hence do not get returned; we then get
        keyErrors when trying to compare faetures later!
        '''

        #set up dict first!
        var_hits = {}
        for var in (self.running_vars):
            var_hits[var] = 0

        for n_iter in range(self.n_trainings):
            #create shadow set with remaining features
            x_mirror = self.create_shadow()
            print ('training classifier for iteration: {}'.format(n_iter))
            clf = xg.XGBClassifier(objective='binary:logistic', **(self.train_params))
            clf.fit(x_mirror, self.y_train, sample_weight=self.w_train)
            print ('done')

            #n_importance = clf.get_booster().get_score(importance_type='gain')
            explainer = shap.Explainer(clf)
            shap_values = explainer(x_mirror)
            vals = np.abs(shap_values.values).mean(0)
            n_importance = {key:value for (key,value) in zip(x_mirror.columns,vals)}


            best_shadow_imp = 0
            for var in self.running_shadow_vars:
                if var in n_importance.keys():
                    if n_importance[var] > best_shadow_imp: 
                        best_shadow_imp = n_importance[var]

            #update importances
            for var in self.running_vars:
                if var in n_importance.keys(): 
                    if n_importance[var] > best_shadow_imp: var_hits[var] += 1

        print (var_hits)

        return var_hits


    def check_stopping_criteria(self, removed_vars) -> list[int]:
        '''
        check various stopping criteria e.g. how many features have been removed?, ...
        '''

        if len(removed_vars) >= self.max_vars_removed: return True
        else: return False


    def slim_features(self) -> list[str]:
        '''
        Execute the slimming algorithm
        '''

        print ('starting Boruta algorithm')
        removed_vars     = []
        for i_iter in range(self.i_iters):
            print ('running slimming iteration {}'.format(i_iter))
            var_hits = self.run_trainings()

            #use hits to remove features
            new_running_vars = []
            for real_feature in self.running_vars:
                hits = var_hits[real_feature]
                if hits > self.hit_threshold:
                    new_running_vars.append(real_feature)
                else:
                    print ('Removing feature: {} !'.format(real_feature))
                    removed_vars.append(real_feature)
            #update running vars for real and shadow
            self.update_vars(new_running_vars)
            if(self.check_stopping_criteria(removed_vars)): break
        print ('Boruta removed variables: {}'.format(removed_vars))

        #return optimal set of vars
        print ('final kept variables are: {}'.format(self.running_vars))
        return self.running_vars 


    def __call__(self):
        return self.slim_features()
