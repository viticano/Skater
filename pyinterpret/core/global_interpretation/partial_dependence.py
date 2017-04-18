"""Partial Dependence class"""

from itertools import product, cycle
import numpy as np
import pandas as pd
import re
from pathos.multiprocessing import Pool
import functools
from six.moves import copyreg as copy_reg
import types

from .base import BaseGlobalInterpretation
from ...model.model import ModelType
from ...util import exceptions
from ...util.data_structures import ControlledDict
from ...util.model import get_predictor
from ...util.kernels import flatten
from ...util.plotting import if_matplotlib,COLORS, ColorMap, \
    coordinate_gradients_to_1d_colorscale, plot_2d_color_scale

#if we want to employ instance methods in multiprocessing, enable this code:
#copy_reg.pickle(types.MethodType, pickle_method, unpickle_method)
#methods stored in util.serialization

def _compute_pd(index, estimator_fn, grid_expanded, pd_metadata, input_data):
    """ Helper function to compute partial dependence for each grid value

    Parameters:
    -----------
    index(int): row index for the grid
    estimator_fn(estimator.function):
        an estimator function of a fitted model used to derive prediction.
        Supports classification(binary, multi-class) and regression.
    grid_expanded(numpy.ndarray:
        The grid of ``target_labels` for which partial dependence needs to be computed
    number_of_classes(int):
        unique number of classes in the ``target_labels``
    feature_ids(list):
        the names/ids of the features for which partial dependence is to be computed.
    input_data(numpy.ndarray):
        input sample data as array to compute partial dependence

    Returns
    -------
    pd_dict(dict, shape={'sd': <>, 'val_1': <>, 'mean'} : containing estimated value on sample dataset
    """
    #print "process id: {} parent process id {} \n".format(*[os.getpid(), os.getppid()])
    feature_columns = pd_metadata['feature_columns']
    feature_ids= pd_metadata['feature_ids']
    class_columns = pd_metadata['class_columns']
    data_sample = input_data.copy()
    new_row = grid_expanded[index]

    for feature_idx, feature_id in enumerate(feature_ids):
        data_sample[feature_id] = new_row[feature_idx]

    predictions = estimator_fn(data_sample.values)
    mean_prediction = np.mean(predictions, axis=0)
    std_prediction = np.std(predictions, axis=0)

    if len(predictions.shape)==1:
        mean_prediction = np.array([mean_prediction])
        std_prediction = np.array([std_prediction])

    number_of_classes = len(class_columns)
    assert number_of_classes == mean_prediction.shape[0], "Mismatch between prediction dimension" \
                                                          "and number of class names" \
                                                          "Prediction dimension: {} " \
                                                          "n class_names: {}".format(*[mean_prediction.shape,
                                                                                       number_of_classes])

    pd_dict = {column: new_row[idx] for idx, column in enumerate(feature_columns)}
    if number_of_classes == 2:
        class_column = class_columns[1]
        pd_dict[class_column] = mean_prediction[1]
        pd_dict['sd'] = std_prediction[0]
    else:
        for class_i in range(number_of_classes):
            pd_dict[class_columns[class_i]] = mean_prediction[class_i]
        pd_dict['sd'] = std_prediction[0]

    return pd_dict


class PartialDependence(BaseGlobalInterpretation):
    """Contains methods for partial dependence. Subclass of BaseGlobalInterpretation"""

    __all__ = ['partial_dependence', 'plot_partial_dependence']

    _pdp_metadata = {}
    _predict_fn = None


    def _build_metadata_dict(self, modelinstance, feature_ids):

        feature_columns = ['feature: {}'.format(i) for i in feature_ids]
        sd_col = 'sd'
        class_names = modelinstance.class_names
        metadata = ControlledDict({
            'sd_column': sd_col,
            'class_columns': class_names,
            'feature_columns': feature_columns,
            'feature_ids':feature_ids
        })
        metadata.block_setitem()
        return metadata


    def partial_dependence(self, feature_ids, modelinstance, grid=None, grid_resolution=None,
                           n_jobs=-1, grid_range=None, sample=True,
                           sampling_strategy='random-choice', n_samples=10000,
                           bin_count=50, samples_per_bin=10, return_metadata=False):

        """
        Computes partial_dependence of a set of variables. Essentially approximates
        the partial partial_dependence of the predict_fn with respect to the variables
        passed.

        Parameters:
        -----------
        feature_ids: list
            the names/ids of the features for which partial dependence is to be computed.
            Note that the algorithm's complexity scales exponentially with additional
            features, so generally one should only look at one or two features at a
            time. These feature ids must be available in the class's associated DataSet.
            As of now, we only support looking at 1 or 2 features at a time.
        modelinstance: pyinterpret.model.model.Model subtype
            an estimator function of a fitted model used to derive prediction. Supports
            classification and regression. Supports classification(binary, multi-class) and regression.
            predictions = predict_fn(data)

            Can either by a pyinterpret.model.remote.DeployedModel or a
            pyinterpret.model.local.InMemoryModel
        grid: numpy.ndarray
            2 dimensional array on which we fix values of features. Note this is
            determined automatically if not given based on the percentiles of the
            dataset.
        grid_resolution: int
            how many unique values to include in the grid. If the percentile range
            is 5% to 95%, then that range will be cut into <grid_resolution>
            equally size bins. Defaults to 100 for 1D and 30 for 2D.
        n_jobs: int
            The number of CPUs to use to compute the PDs. -1 means 'all CPUs'.
            Defaults to using all cores(-1).
        grid_range: tuple
            the percentile extrama to consider. 2 element tuple, increasing, bounded
            between 0 and 1.
        sample: boolean
            Whether to sample from the original dataset.
        sampling_strategy: string
            If sampling, which approach to take. See DataSet.generate_sample for
            details.
        n_samples: int
            The number of samples to use from the original dataset. Note this is
            only active if sample = True and sampling strategy = 'uniform'. If
            using 'uniform-over-similarity-ranks', use samples per bin
        bin_count: int
            The number of bins to use when using the similarity based sampler. Note
            this is only active if sample = True and
            sampling_strategy = 'uniform-over-similarity-ranks'.
            total samples = bin_count * samples per bin.
        samples_per_bin: int
            The number of samples to collect for each bin within the sampler. Note
            this is only active if sample = True and
            sampling_strategy = 'uniform-over-similarity-ranks'. If using
            sampling_strategy = 'uniform', use n_samples.
            total samples = bin_count * samples per bin.
        return_metadata: boolean

        Example
        --------
        >>> from sklearn.ensemble import RandomForestClassier
        >>> rf = RandomForestClassier()
        >>> rf.fit(X,y)
        >>> partial_dependence(feature_ids, rf.predict)
        >>> partial_dependence(feature_ids, rf.predict_proba)
        """

        if not hasattr(feature_ids, "__iter__"):
            feature_ids = [feature_ids]

        if grid_resolution is None:
            grid_resolution = 100 if len(feature_ids) == 1 else 30

        # TODO: There might be a better place to do this check
        if not isinstance(modelinstance, ModelType):
            raise(exceptions.ModelError("Incorrect estimator function used for computing partial dependence, try one "
                                        "creating one with pyinterpret.model.local.InMemoryModel or"
                                        "pyinterpret.model.remote.DeployedModel"))

        if modelinstance.model_type == 'classifier' and modelinstance.probability == False:
            raise(exceptions.ModelError("Incorrect estimator function used for computing partial dependence, try one "
                                        "with which give probability estimates"))

        if len(feature_ids) >= 3:
            too_many_features_err_msg = "Pass in at most 2 features for pdp. If you have a " \
                                        "use case where you'd like to look at 3 simultaneously" \
                                        ", please let us know."
            raise(exceptions.TooManyFeaturesError(too_many_features_err_msg))

        if len(feature_ids) == 0:
            empty_features_err_msg = "Feature ids must have non-zero length"
            raise(exceptions.EmptyFeatureListError(empty_features_err_msg))

        if len(set(feature_ids)) != len(feature_ids):
            duplicate_features_error_msg = "feature_ids cannot contain duplicate values"
            raise(exceptions.DuplicateFeaturesError(duplicate_features_error_msg))

        if self.data_set is None:
            load_data_not_called_err_msg = "self.interpreter.data_set not found. " \
                                           "Please call Interpretation.load_data " \
                                           "before running this method."
            raise(exceptions.DataSetNotLoadedError(load_data_not_called_err_msg))

        # TODO: This we can change easily to functional style
        missing_feature_ids = []
        for feature_id in feature_ids:
            if feature_id not in self.data_set.feature_ids:
                missing_feature_ids.append(feature_id)

        if missing_feature_ids:
            missing_feature_id_err_msg = "Features {0} not found in " \
                                         "Interpretation.data_set.feature_ids" \
                                         "{1}".format(missing_feature_ids, self.data_set.feature_ids)
            raise(KeyError(missing_feature_id_err_msg))

        if grid_range is None:
            grid_range = (.05, 0.95)
        else:
            if not hasattr(grid_range, "__iter__"):
                err_msg = "Grid range {} needs to be an iterable".format(grid_range)
                raise(exceptions.MalformedGridRangeError(err_msg))

        self._check_grid_range(grid_range)

        # if you dont pass a grid, build one.
        grid = np.array(grid)
        if not grid.any():
            # Currently, if a given feature only has two unique values
            # then the grid will only include those two. Otherwise itll take the percentile
            # range according with grid_resolution bins.
            # sklearn however just returns the all unique values if the number of unique
            # values is less then grid resolution.
            # TODO: evaluate cases when len(unique(feature))==2
            grid = self.data_set.generate_grid(feature_ids,
                                               grid_resolution=grid_resolution,
                                               grid_range=grid_range)
        else:
            if len(grid.shape) == 1 and not hasattr(grid[0], "__iter__"):
                grid = grid[:, np.newaxis].T
                grid_resolution = grid.shape[1]

        self.interpreter.logger.debug("Grid shape used for pdp: {}".format(grid.shape))
        self.interpreter.logger.debug("Grid resolution for pdp: {}".format(grid_resolution))

        # make sure data_set module is giving us correct data structure
        self._check_grid(grid, feature_ids)

        # generate data
        data_sample = self.data_set.generate_sample(strategy=sampling_strategy,
                                                    sample=sample,
                                                    n_samples_from_dataset=n_samples,
                                                    samples_per_bin=samples_per_bin,
                                                    bin_count=bin_count)

        self.interpreter.logger.debug("Shape of sampled data: {}".format(data_sample.shape))
        #TODO: Add check for non-empty data

        # make sure data_set module is giving us correct data structure
        self._check_dataset_type(data_sample)

        _pdp_metadata = self._build_metadata_dict(modelinstance, feature_ids)
        self.interpreter.logger.debug("PD metadata: {}".format(_pdp_metadata))

        # cartesian product of grid
        grid_expanded = np.array(list(product(*grid)))

        if grid_expanded.shape[0] <= 0:
            empty_grid_expanded_err_msg = "Must have at least 1 pdp value" \
                                          "grid shape: {}".format(grid_expanded.shape)
            raise(exceptions.MalformedGridError(empty_grid_expanded_err_msg))

        n_classes = modelinstance.n_classes
        pd_list = []

        predict_fn = get_predictor(modelinstance)

        n_jobs = None if n_jobs < 0 else n_jobs
        pd_func = functools.partial(_compute_pd,
                                    estimator_fn=predict_fn,
                                    grid_expanded=grid_expanded,
                                    pd_metadata=_pdp_metadata,
                                    input_data=data_sample)
        arg_list = [i for i in range(grid_expanded.shape[0])]
        executor_instance = Pool(n_jobs)

        try:
            pd_list = executor_instance.map(pd_func, arg_list)
        except:
            pd_list = map(pd_func, arg_list)
        finally:
            executor_instance.close()
            executor_instance.join()
            executor_instance.terminate()
        if return_metadata:
            return pd.DataFrame(pd_list), _pdp_metadata
        else:
            return pd.DataFrame(pd_list)

    @if_matplotlib
    def plot_partial_dependence(self, feature_ids, modelinstance, grid=None,
                                grid_resolution=None, grid_range=None, n_jobs=-1,
                                sample=True, sampling_strategy='random-choice',
                                n_samples=10000, bin_count=50, samples_per_bin=10,
                                with_variance=False):
        """
        Computes partial_dependence of a set of variables. Essentially approximates
        the partial partial_dependence of the predict_fn with respect to the variables
        passed.

        Parameters:
        -----------
        feature_ids: list
            the names/ids of the features for which partial dependence is to be computed.
            Note that the algorithm's complexity scales exponentially with additional
            features, so generally one should only look at one or two features at a
            time. These feature ids must be available in the class's associated DataSet.
            As of now, we only support looking at 1 or 2 features at a time.
        modelinstance: pyinterpret.model.model.Model subtype
            an estimator function of a fitted model used to derive prediction. Supports
            classification and regression. Supports classification(binary, multi-class) and regression.
            predictions = predict_fn(data)

            Can either by a pyinterpret.model.remote.DeployedModel or a
            pyinterpret.model.local.InMemoryModel
        grid: numpy.ndarray
            2 dimensional array on which we fix values of features. Note this is
            determined automatically if not given based on the percentiles of the
            dataset.
        grid_resolution: int
            how many unique values to include in the grid. If the percentile range
            is 5% to 95%, then that range will be cut into <grid_resolution>
            equally size bins. Defaults to 100 for 1D and 30 for 2D.
        grid_range: tuple
            the percentile extrama to consider. 2 element tuple, increasing, bounded
            between 0 and 1.
        n_jobs: int
            The number of CPUs to use to compute the PDs. -1 means 'all CPUs'.
            Defaults to using all cores(-1).
        sample: boolean
            Whether to sample from the original dataset.
        sampling_strategy: string
            If sampling, which approach to take. See DataSet.generate_sample for
            details.
        n_samples: int
            The number of samples to use from the original dataset. Note this is
            only active if sample = True and sampling strategy = 'uniform'. If
            using 'uniform-over-similarity-ranks', use samples per bin
        bin_count: int
            The number of bins to use when using the similarity based sampler. Note
            this is only active if sample = True and
            sampling_strategy = 'uniform-over-similarity-ranks'.
            total samples = bin_count * samples per bin.
        samples_per_bin: int
            The number of samples to collect for each bin within the sampler. Note
            this is only active if sample = True and
            sampling_strategy = 'uniform-over-similarity-ranks'. If using
            sampling_strategy = 'uniform', use n_samples.
            total samples = bin_count * samples per bin.


        with_variance(bool):
            whether to include pdp error bars in the plots. Currently disabled for 3D
            plots for visibility. If you have a use case where you'd like error bars for
            3D pdp plots, let us know!
        plot_title(string):
            title for pdp plots

        Example
        --------
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> from sklearn.datasets.california_housing import fetch_california_housing
        >>> cal_housing = fetch_california_housing()
        # split 80/20 train-test
        >>> x_train, x_test, y_train, y_test = train_test_split(cal_housing.data,
        >>>                             cal_housing.target, test_size=0.2, random_state=1)
        >>> names = cal_housing.feature_names
        >>> print("Training the estimator...")
        >>> estimator = GradientBoostingRegressor(n_estimators=10, max_depth=4,
        >>>                             learning_rate=0.1, loss='huber', random_state=1)
        >>> estimator.fit(x_train, y_train)
        >>> from pyinterpret.core.explanations import Interpretation
        >>> interpreter = Interpretation()
        >>> print("Feature name: {}".format(names))
        >>> interpreter.load_data(X_train, feature_names=names)
        >>> print("Input feature name: {}".format[names[1], names[5]])
        >>> from pyinterpret.model import InMemoryModel
        >>> model = InMemoryModel(clf.predict, examples = X_train)
        >>> interpreter.partial_dependence.plot_partial_dependence([names[1], names[5]], model,
        >>>                                                         n_samples=100, n_jobs=1)

        """

        # in the event that a user wants a 3D pdp with multiple classes, how should
        # we handle this? currently each class will get its own figure
        if not hasattr(feature_ids, "__iter__"):
            pd_df, metadata = self.partial_dependence(feature_ids, modelinstance, grid=grid,
                                                      grid_resolution=grid_resolution,
                                                      grid_range=grid_range, sample=sample,
                                                      sampling_strategy=sampling_strategy,
                                                      n_samples=n_samples, bin_count=bin_count,
                                                      samples_per_bin=samples_per_bin,
                                                      n_jobs=n_jobs, return_metadata=True)

            self.interpreter.logger.info("done computing pd, now plotting ...")
            ax = self._plot_pdp_from_df(pd_df, metadata, with_variance=with_variance)
            return ax
        else:
            ax_list = []
            for feature_or_feature_pair in feature_ids:
                pd_df, metadata = self.partial_dependence(feature_or_feature_pair, modelinstance,
                                                          grid=grid,
                                                          grid_resolution=grid_resolution,
                                                          grid_range=grid_range, sample=sample,
                                                          sampling_strategy=sampling_strategy,
                                                          n_samples=n_samples, bin_count=bin_count,
                                                          samples_per_bin=samples_per_bin,
                                                          n_jobs=n_jobs, return_metadata=True)

                self.interpreter.logger.info("done computing pd, now plotting ...")
                ax = self._plot_pdp_from_df(pd_df, metadata, with_variance=with_variance)
                ax_list.append(ax)
            return ax_list

    @if_matplotlib
    def _plot_pdp_from_df(self, pdp, pd_metadata,
                          with_variance=False, plot_title=None,
                          disable_offset=True):

        feature_columns = pd_metadata['feature_columns']
        class_columns = pd_metadata['class_columns']
        sd_col = pd_metadata['sd_column']
        n_features = len(feature_columns)
        if n_features == 1 or not hasattr(feature_columns, "__iter__"):
            feature_column = feature_columns[0]
            return self._2d_pdp_plot(pdp, feature_column, sd_col, class_columns,
                                     with_variance=with_variance,
                                     plot_title=plot_title, disable_offset=disable_offset)
        elif n_features == 2:
            feature1_column, feature2_column = feature_columns
            return self._3d_pdp_plot(pdp, feature1_column, feature2_column,
                                     sd_col, class_columns, with_variance=with_variance,
                                     plot_title=plot_title)
        else:
            msg = "Something went wrong. Expected either a single feature, " \
                  "or a 1-2 element array of features, got array of size:" \
                  "{}: {}".format(*[n_features, feature_columns])
            raise(ValueError(msg))

    @if_matplotlib
    def _2d_pdp_plot(self, pdp, feature_name, sd_col, class_columns,
                     with_variance=False, plot_title=None, disable_offset=True):
        colors = cycle(COLORS)
        figure_list, axis_list = [], []

        # if there are just 2 classes, pick the last one.
        if len(class_columns) == 2:
            class_columns = [class_columns[-1]]

        for class_column in class_columns:
            # if class_name is None:
            #     raise ValueError("Could not parse class name from {}".format(mean_col))
            f, ax = plt.subplots(1)
            figure_list.append(f)
            axis_list.append(ax)
            color = next(colors)

            data = pdp.set_index(feature_name)
            plane = data[class_column]

            # if binary feature, then len(pdp) == 2 -> barchart
            if self._is_feature_binary(pdp, feature_name):
                if with_variance:
                    error = data[sd_col]
                else:
                    error = None
                plane.plot(kind='bar', ax=ax, color=color, yerr=error)
            else:
                plane.plot(ax=ax, color=color)
                if with_variance:
                    upper_plane = plane + data[sd_col]
                    lower_plane = plane - data[sd_col]
                    ax.fill_between(data.index.values,
                                    lower_plane.values,
                                    upper_plane.values,
                                    alpha=.2,
                                    color=color)

            if plot_title:
                ax.set_title(plot_title)
            ax.set_ylabel(class_column)
            ax.set_xlabel(feature_name)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
            if disable_offset:
                ax.yaxis.set_major_formatter(ScalarFormatter())
        return flatten([figure_list, axis_list])


    def _is_feature_binary(self, pdp, feature):
        data = pdp[feature].values
        if len(np.unique(data)) == 2:
            return True
        else:
            return False

    @if_matplotlib
    def _3d_pdp_plot(self, pdp, feature1, feature2, sd_column, class_columns,

                     with_variance=False, plot_title=None, disable_offset=True):

        # if there are just 2 classes, pick the last one.
        if len(class_columns) == 2:
            class_columns = [class_columns[-1]]

        feature_1_data = pdp[feature1].values
        feature_2_data = pdp[feature2].values

        feature_1_is_binary = len(np.unique(feature_1_data)) == 2
        feature_2_is_binary = len(np.unique(feature_2_data)) == 2

        if not feature_1_is_binary and not feature_2_is_binary:
            plot_objects = if_matplotlib(self._plot_3d_full_mesh(pdp, feature1, feature2,
                                                   sd_column, class_columns,
                                                   with_variance=with_variance))

        elif feature_1_is_binary and feature_2_is_binary:
            plot_objects = self._plot_2d_2_binary_feature(pdp,
                                                   feature1,
                                                   feature2,
                                                   sd_column, class_columns,
                                                   with_variance=with_variance)
        else:
            # one feature is binary and one isnt.
            binary_feature, non_binary_feature = {
                feature_1_is_binary: [feature1, feature2],
                (not feature_1_is_binary): [feature2, feature1]
            }[feature_1_is_binary]

            plot_objects = self._plot_2d_1_binary_feature_and_1_continuous(pdp,
                                                   binary_feature,
                                                   non_binary_feature,
                                                   sd_column, class_columns,
                                                   with_variance=with_variance)
        for obj in plot_objects:
            if isinstance(obj, mpl_axes):
                if disable_offset:
                    obj.xaxis.set_major_formatter(ScalarFormatter())
                    obj.yaxis.set_major_formatter(ScalarFormatter())
                if plot_title:
                    obj.set_title("Partial Dependence")
                # matplotlib increases x from left to right, flipping that
                # so the origin is front and center
        return plot_objects

    @if_matplotlib
    def _plot_3d_full_mesh(self, pdp, feature1, feature2,
                           sd_column, class_columns,
                           with_variance=False, alpha=.7):
        colors = cycle(COLORS)

        figure_list, axis_list = [], []

        for class_column in class_columns:
            gradient_x, gradient_y, X, Y, Z = self.compute_3d_gradients(pdp, class_column, feature1, feature2)
            color_gradient, xmin, xmax, ymin, ymax = coordinate_gradients_to_1d_colorscale(gradient_x, gradient_y)
            plt.figure()
            ax = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3, projection='3d')
            figure_list.append(ax.figure)
            axis_list.append(ax)

            surface = ax.plot_surface(X, Y, Z, alpha=alpha, facecolors=color_gradient, linewidth=0., rstride=1, cstride=1)
            dx_mean = np.mean(gradient_x)
            dy_mean = np.mean(gradient_y)
            mean_point = (dx_mean, dy_mean)

            #add 2D color scale
            ax_colors = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=1)
            ax_colors = plot_2d_color_scale(xmin, xmax, ymin, ymax, plot_point=mean_point,ax=ax_colors)
            ax_colors.set_xlabel("Local Impact {}".format(feature1))
            ax_colors.set_ylabel("Local Impact {}".format(feature2))


            if with_variance:
                var_color = next(colors)
                ax.plot_trisurf(pdp[feature1].values, pdp[feature2].values,
                                (pdp[class_column] + pdp[sd_column]).values, alpha=.2,
                                color=var_color)
                ax.plot_trisurf(pdp[feature1].values, pdp[feature2].values,
                                (pdp[class_column] - pdp[sd_column]).values, alpha=.2,
                                color=var_color)
            ax.set_xlabel(feature1)
            ax.set_ylabel(feature2)
            ax.set_zlabel(class_column)
            ax.invert_xaxis()

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)

        return flatten([figure_list, axis_list])

    @if_matplotlib
    def _plot_3d_2_binary_feature(self, pdp, feature1, feature2, sd_column,
                                  class_columns, with_variance=False):
        colors = cycle(COLORS)
        figure_list, axis_list = [], []
        for class_column in class_columns:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for val in np.unique(pdp[feature2]):
                filter_idx = pdp[feature2] == val
                pdp_vals = pdp[filter_idx][class_column].values
                x1 = pdp[filter_idx][feature1].values
                x2 = pdp[filter_idx][feature2].values
                ax.plot(x1, x2, pdp_vals)

            figure_list.append(fig)
            axis_list.append(ax)
            color = next(colors)
            ax.set_xlabel(feature1)
            ax.set_ylabel(feature2)
            ax.set_zlabel(class_column)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
        return flatten([figure_list, axis_list])

    def _plot_2d_2_binary_feature(self, pdp, feature1, feature2, sd_col,
                                  class_columns, with_variance=False):
        figure_list, axis_list = [], []

        std_error = pdp.set_index([feature1, feature2])[sd_col].unstack()
        for class_column in class_columns:
            f = plt.figure()
            ax = f.add_subplot(111)
            #feature2 is columns
            #feature1 is index
            plot_data = pdp.set_index([feature1, feature2])[class_column].unstack()
            plot_data.plot(ax=ax, color=COLORS)

            if with_variance:
                colors = cycle(COLORS)
                binary1_values = plot_data.index.values
                binary2_values = plot_data.columns.values
                for binary2_value in binary2_values:
                    color = next(colors)
                    yerr = std_error[binary2_value].values
                    upper_plane = yerr + plot_data[binary2_value].values
                    lower_plane = plot_data[binary2_value].values - yerr
                    ax.fill_between(binary1_values, lower_plane, upper_plane,
                                    color=color,alpha=.2)
            figure_list.append(f)
            axis_list.append(ax)
            ax.set_xlabel(feature1)
            ax.set_ylabel(class_column)

        return flatten([figure_list, axis_list])

    def _plot_2d_1_binary_feature_and_1_continuous(self, pdp, binary_feature,
                                                   non_binary_feature, sd_column,
                                                   class_columns, with_variance=False):

        figure_list, axis_list = [], []

        binary_vals = np.unique(pdp[binary_feature])
        for class_column in class_columns:
            colors = cycle(COLORS)
            f = plt.figure()
            ax = f.add_subplot(111)
            figure_list.append(f)
            axis_list.append(ax)
            plot_data = pdp.set_index([non_binary_feature, binary_feature])[class_column]\
                .unstack()
            sd = pdp.set_index([non_binary_feature, binary_feature])[sd_column]\
                .unstack()

            plot_data.plot(ax=ax,color=COLORS)
            if with_variance:
                non_binary_values = plot_data.index.values
                binary_values = plot_data.columns.values
                upper_plane = plot_data + sd
                lower_plane = plot_data - sd
                for binary_value in binary_values:
                    color = next(colors)
                    ax.fill_between(non_binary_values, lower_plane[binary_value].values, upper_plane[binary_value].values, alpha=.2,
                                    color=color)
            ax.set_ylabel(class_column)
        return flatten([figure_list, axis_list])

    @staticmethod
    def _check_grid(grid, feature_ids):
        if not isinstance(grid, np.ndarray):
            err_msg = "Grid of type {} must be a numpy array".format(type(grid))
            raise(exceptions.MalformedGridError(err_msg))

        if len(feature_ids) != grid.shape[0]:
            err_msg = "Given {0} features, there must be {1} rows in grid" \
                      "but {2} were found".format(len(feature_ids),
                                                  len(feature_ids),
                                                  grid.shape[0])
            raise(exceptions.MalformedGridError(err_msg))


    @staticmethod
    def _check_dataset_type(dataset):
        """
        Ensures that dataset is pandas dataframe
        :param dataset:
        :return:
        """
        if not isinstance(dataset, pd.DataFrame):
            err_msg = "Dataset.data must be a pandas.dataframe"
            raise(exceptions.DataSetError(err_msg))


    @staticmethod
    def _check_grid_range(grid_range):
        """
        Tested by unit test, ensures grid range is between 0 and 1
        :param grid_range (tuple)

        """
        if len(grid_range) != 2:
            err_msg = "Grid range {} must have 2 elements".format(grid_range)
            raise(exceptions.MalformedGridRangeError(err_msg))
        if not all([i >= 0 and i <= 1 for i in grid_range]):
            err_msg = "All elements of grid range {} " \
                      "must be between 0 and 1".format(grid_range)
            raise(exceptions.MalformedGridRangeError(err_msg))

    @staticmethod
    def compute_3d_gradients(pdp, mean_col, feature_1, feature_2, scaled=True):
        """
        Computes component-wise gradients of pdp dataframe.
        
        Parameters
        ----------
        pdp: pandas.DataFrame
            DataFrame containing partial dependence values
        mean_col: string
            column name corresponding to pdp value
        feature_1: string
            column name corresponding to feature 1
        feature_2: string
            column name corresponding to feature 2
        scaled: bool
            Whether to scale the x1 and x2 gradients relative to x1 and x2 bin sizes

        Returns
        ----------
        dx, dy, x_matrix, y_matrix, z_matrix
        """
        def feature_vals_to_grad_deltas(values):
            values = np.unique(values)
            values.sort()
            diffs = np.diff(values)
            conv_diffs = np.array([(diffs[i] + diffs[i + 1]) / 2 for i in range(0, len(diffs) - 1)])
            diffs = np.concatenate((np.array([diffs[0]]), conv_diffs, np.array([diffs[-1]])))
            return diffs

        df = pdp.sort(columns=[feature_1, feature_2])

        feature_1_diffs = feature_vals_to_grad_deltas(df[feature_1].values)
        feature_2_diffs = feature_vals_to_grad_deltas(df[feature_2].values)

        x1_size = feature_1_diffs.shape[0]
        x2_size = feature_2_diffs.shape[0]

        z_matrix = np.zeros((x1_size, x2_size))
        x1_matrix = np.zeros((x1_size, x2_size))
        x2_matrix = np.zeros((x1_size, x2_size))

        for i in range(x1_size):
            for j in range(x2_size):
                idx = i * x2_size + j
                x1_val = df[feature_1].iloc[idx]
                x2_val = df[feature_2].iloc[idx]
                z_val = df[mean_col].iloc[idx]

                z_matrix[i, j] = z_val
                x1_matrix[i, j] = x1_val
                x2_matrix[i, j] = x2_val

        dx1, dx2 = np.gradient(z_matrix)
        if scaled:
            dx1 = np.apply_along_axis(lambda x: x / feature_1_diffs, 0, dx1)
            dx2 = np.apply_along_axis(lambda x: x / feature_2_diffs, 1, dx2)
        return dx1, dx2, x1_matrix, x2_matrix, z_matrix
