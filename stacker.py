import numpy as np

class stacking_ensemble(object):

    def __init__(self, base_models, stacking_model, k_folds=3, parameter_grids=None):

        # 'models' will be a list of base model.
        self.base_models = base_models
        self.stacking_model = stacking_model
        self.k_folds = k_folds

        # Booleans for telling if the top model functions have been used
        self.top_models_gs = False
        self.top_models_non_gs = False

        self.parameter_grids = parameter_grids

    def find_top_models(self, X, y, k_models=1, cv=None, subsample=None, scoring=None):
        """
        DOCSTRING

        This function will look at the basemodels in, and finds the best models and then saves the best
        model/parameter combinations to a new object which can be used for fitting.

        scoring: way of comparing performance of models.  Can take on all values from SKLearn model
        evaluation documentation

        subsample: downsample the training data to make model selection faster.  Values between 0 and 1.
        """

        X_data = np.array(X)
        y_data = np.array(y)

        if subsample:
            axes_choices = np.random.binomial(n=1, p=subsample, size=X.shape[0])
            X_data = X_data[axes_choices == 1]
            y_data = y_data[axes_choices == 1]

        best_models = []
        from sklearn.model_selection import cross_val_score

        model_scores = []
        for model in self.base_models:
            score = np.mean(cross_val_score(model, X_data, y_data, scoring=scoring, cv=cv))
            model_scores.append({'model': model, 'score': score})

        self.top_models = sorted(model_scores, key=lambda k: k['score'], reverse=True)[:k_models]
        self.top_models_non_gs = True

    def find_top_models_gs(self, X, y, k_models=1, cv=None, parameter_grids=None, subsample=None,
                           scoring=None, verbose=0):
        """
        This will find the top models by gridsearching.

        To use this feature, base_models and parameter grids must be passed in the same order
        Parameter grids need to be a list of dictionaries of sklearn parameters

        """
        # set class variable parameter grids to this function's version if it hasn't been set yet or this one is passed
        if not self.parameter_grids or parameter_grids:
            self.parameter_grids = parameter_grids

        from sklearn.model_selection import GridSearchCV

        model_scores = []

        X_data = np.array(X)
        y_data = np.array(y)

        if subsample:
            axes_choices = np.random.binomial(n=1, p=subsample, size=X.shape[0])
            X_data = X_data[axes_choices == 1]
            y_data = y_data[axes_choices == 1]

        for model, parameters in zip(self.base_models, self.parameter_grids):

            gscv = GridSearchCV(model, param_grid=parameters, cv=cv, verbose=verbose)
            gscv.fit(X_data, y_data)

            model_scores += ([{'model': model,
                   'score': score,
                   'parameters': params} for params, score in zip(gscv.cv_results_['params'],
                                                               gscv.cv_results_['mean_test_score'])])

        self.top_models = sorted(model_scores, key=lambda k: k['score'], reverse=True)[:k_models]
        self.top_models_gs = True

    def fit(self, X, y, use_top=True):
        X = np.array(X)
        y = np.array(y)

        df_meta = np.zeros(X.shape)
        fold_ids = np.zeros(df_meta.shape[0])

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.k_folds, shuffle=True)

        for i, (train, test) in enumerate(kf.split(X, y)):
            fold_ids[test] = i


        if (not self.top_models_gs and not self.top_models_non_gs) or not use_top:

            print 'fitting base models'
            for k in set(fold_ids):
                print '---> Fitting fold ', k + 1
                train_fold = X[fold_ids != k, :]

                for i, model in enumerate(self.base_models):
                    print '------> Fitting model ', str(i)
                    model.fit(train_fold, y[fold_ids != k])
                    df_meta[fold_ids == k, i] = model.predict(X[fold_ids == k, :])

            # Fit each base model to the entire X training set once we've fit the meta columns
            for model in self.base_models:
                model.fit(X, y)

        else:
            # If we've grid searched best models, here we'll fit the parameters to all the models.
            if self.top_models_gs:
                for model_dict in self.top_models:
                    model_dict['model'].set_params(**model_dict['parameters'])

            for k in set(fold_ids):
                print '---> Fitting fold ', k + 1
                train_fold = X[fold_ids != k, :]

                for i, model_dict in enumerate(self.top_models):
                    print '------> Fitting model ', str(i)
                    model_dict['model'].fit(train_fold, y[fold_ids != k])
                    df_meta[fold_ids == k, i] = model_dict['model'].predict(X[fold_ids == k, :])

            # Fit each base model to the entire X training set once we've fit the meta columns
            for model_dict in self.top_models:
                model_dict['model'].fit(X, y)

        # FITTING THE FINAL META MODEL
        print 'fitting stacking model...'
        self.stacking_model.fit(df_meta, y)
        print 'done!'

    def predict(self, X):
        X = np.array(X)
        test_meta = np.zeros(X.shape)

        if not self.top_models_gs and not self.top_models_non_gs:
            for i, model in enumerate(self.base_models):
                test_meta[:, i] = model.predict(X)
        else:
            for i, model_dict in enumerate(self.top_models):
                test_meta[:, i] = model_dict['model'].predict(X)
        predictions = self.stacking_model.predict(test_meta)
        return predictions
