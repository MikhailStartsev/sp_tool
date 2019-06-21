#!/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np
import abc
import copy

from arff_helper import ArffHelper
from recording_processor import RecordingProcessor


class SmoothPursuitDetector(object):
    """
    DBSCAN-based smooth pursuit detector. All the logic is in respective DBSCANWith* classes, this is just a wrapper
    that based on the arguments to __init__ method chooses either DBSCANWithMinPts or  DBSCANWithMinObservers.
    """
    def __init__(self, eps_deg=2.0, time_slice_microsec=40000,
                 min_pts=None, min_observers=None):
        """
        Initialize the SmoothPursuitDetector object by choosing one of the clustering algorithms and storing it into
        self.clustering. If neither @min_pts nor @min_observers is specified, @min_pts='num_observers' is used.

        :param eps_deg: Spatial Euclidean distance threshold that defines the neighbourhood in the XY-plane.
                        Given in degrees of visual field, a pixel value is assigned when the recordings' data
                        is provided.
        :param time_slice_microsec: Width of the time slice that defines the size of the neighbourhood on the time axis.
                                    Value is given in microseconds. The neighbourhood essentially has cylindrical shape.

        Only one of the following two arguments can be provided (or none, then @min_pts='num_observers' is used).
        :param min_pts: integer indicating the minimum number of points required to
                        form a "valid" neighbourhood (that of a core point).
                        Could also be a 'num_observers' string (default), in which case
                        the actual value is determined during the self._setup_internal_parameters() call
        :param min_observers: either float [0; 1] (indicating the share of all the
                              present observers) or int [2; +\inf) (indicating the
                              absolute threshold for observer count).
        """
        # if neither min_observers, nor min_pts is provided, reset to default
        if min_pts is None and min_observers is None:
            min_pts = 'num_observers'
        # choose one one of the algorithms based on parameters
        assert (min_pts is not None) ^ (min_observers is not None), \
            'Either @min_pts or @min_observers must be provided!'
        if min_pts is not None:
            self.clustering = DBSCANWithMinPts(eps_deg=eps_deg, time_slice_microsec=time_slice_microsec,
                                               min_pts=min_pts)
        else:
            self.clustering = DBSCANWithMinObservers(eps_deg=eps_deg, time_slice_microsec=time_slice_microsec,
                                                     min_observers=min_observers)

    def detect(self, gaze_points_list, inplace=False):
        return self.clustering.cluster(gaze_points_list=gaze_points_list,
                                       inplace=inplace)


class DBSCANWithTimeSlice(object):
    """
    The class is based on DBSCAN algorithm used for density-based data clustering
    (we run this to detect SP, after pre-filtering has removed saccades and fixations).

    Rather than only using spatial locations, the algorithm uses spatio-temporal
    information, i.e. we cluster gaze points data in three-dimensional (t, x, y) space.

    Since there is no a priori optimal scaling factor between time and space,
    we modify the classical DBSCAN notion of the neighbourhood (i.e. a sphere of radius @eps).
    Instead of it, we consider the cylinder with its axis aligned with the time axis.
    This way we have a XY-neighbourhood defined by Euclidean distance and its threshold of @eps,
    and on the temporal axis we take a time slice of @time_slice_microsec width (hence the class name)

    Neighbourhood validation is implemented by two classes that implement the DBSCANWithTimeSlice interface.
    It is done in two different ways, namely "minPts" (validating that the number of other gaze points in the
    neighbourhood is at least @min_pts, closer to original DBSCAN) and "minObservers" (we validate that samples
    of at least @min_observers different observers are present in the neighbourhood).

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, eps_deg=2.0, time_slice_microsec=40000):
        """
        :param eps_deg: Spatial Euclidean distance threshold that defines the neighbourhood in the XY-plane.
                        Given in degrees of visual field, a pixel value is assigned when the recordings' data
                        is provided.
        :param time_slice_microsec: Width of the time slice that defines the size of the neighbourhood on the time axis.
                                    Value is given in microseconds.

        """
        self.time_slice = time_slice_microsec
        self.eps_deg = eps_deg
        self.eps_px = None  # will convert deg to px when data provided

        # initialize empty data
        self._data_set = None
        # store timestamps separately for efficiency
        self._timestamps = None

    def _setup_internal_parameters(self, gaze_points_list):
        """
        This function is used to setup some of the internal parameters that are affected by the data set in use.
        :param gaze_points_list: a list of arff objects (dictionary with 'data' and 'metadata' fields)
        :return:
        """
        ppd = RecordingProcessor.validate_ppd_of_multiple_recordings(gaze_points_list)
        self.eps_px = self.eps_deg * ppd

    def cluster(self, gaze_points_list, inplace=False):
        """
        Find clusters of input gaze data and label clustered points as smooth pursuit.
        Labels (sets the 'EYE_MOVEMENT_TYPE' field) the clusters of data points as 'SP',
        other samples as 'NOISE_CLUSTER'.

        New column 'CLUSTER_ID' is added into the @DATA section of each arff object in @gaze_points_list,
        indicating cluster group ID.

        :param gaze_points_list: a list of arff objects (dictionary with fields such as 'data' and 'metadata')
        :param inplace: whether to modify the original input gaze data with gaze data after clustering or use a copy
        :return: gaze data after clustering in the same form as the input data.

        """
        if not inplace:
            gaze_points_list = copy.deepcopy(gaze_points_list)

        # add global indexing to be able to reference the particular sample even after clustering all in one structure
        for ind in xrange(len(gaze_points_list)):
            ArffHelper.add_column(gaze_points_list[ind], name='global_index', dtype='INTEGER', default_value=-1)
            gaze_points_list[ind]['data']['global_index'] = np.arange(gaze_points_list[ind]['data'].shape[0])

        self._setup_internal_parameters(gaze_points_list)
        self._data_set = self._aggregate_data(gaze_points_list)
        # has to be a copy, so that is is placed continuously in memory
        self._timestamps = self._data_set['time'].copy()

        current_cluster_id = 0

        for i in xrange(len(self._data_set)):
            if self._data_set[i]['visited_flag'] == 1:
                continue
            else:
                self._data_set[i]['visited_flag'] = 1
                neighbourhood = self._get_neighbourhood(i)
                if self._validate_neighbourhood(neighbourhood):
                    # if not: mark current point as NOISE
                    self._expand_cluster(i, neighbourhood, current_cluster_id)
                    current_cluster_id += 1

        # create a new column in gaze_points_list for CLUSTER_ID
        for i in xrange(len(gaze_points_list)):
            ArffHelper.add_column(gaze_points_list[i], 'CLUSTER_ID', 'NUMERIC', -1)

        # label data in gaze_points_list as SP according to CLUSTER_ID
        for i in xrange(len(self._data_set)):
            observer_id = int(self._data_set[i]['observer_id'])
            global_index = self._data_set[i]['global_index']

            if self._data_set[i]['CLUSTER_ID'] != -1:
                gaze_points_list[observer_id]['data']['EYE_MOVEMENT_TYPE'][global_index] = 'SP'
                gaze_points_list[observer_id]['data']['CLUSTER_ID'][global_index] = self._data_set[i]['CLUSTER_ID']
            else:
                gaze_points_list[observer_id]['data']['EYE_MOVEMENT_TYPE'][global_index] = 'NOISE_CLUSTER'

        # can now remove the global_index column
        for ind in xrange(len(gaze_points_list)):
            ArffHelper.remove_column(gaze_points_list[ind], name='global_index')

        return gaze_points_list

    def _expand_cluster(self, current_point, neighbourhood, current_cluster_id):
        """
        Check all points within neighbourhood of current core point in order
        to expand neighbourhood. Processes points in the @self._data_set
        (a 6-column numpy array as data set to be clustered)

        :param current_point: index of the current core point.
        :param neighbourhood: index list as neighbourhood of current core point.
        :param current_cluster_id: index of current cluster.
        :return: index list of expanded neighbourhood points.

        """

        self._data_set[current_point]['CLUSTER_ID'] = current_cluster_id
        for neighbour in neighbourhood:
            if self._data_set[neighbour]['visited_flag'] == 0:
                self._data_set[neighbour]['visited_flag'] = 1
                new_neighbourhood = self._get_neighbourhood(neighbour)  # eps as input parameter
                if self._validate_neighbourhood(new_neighbourhood):
                    new_neighbourhood_set = set(new_neighbourhood)
                    new_neighbours = list(new_neighbourhood_set.difference(neighbourhood))
                    neighbourhood.extend(new_neighbours)    # something wrong if use neighbourhood_set.update

            if self._data_set[neighbour]['CLUSTER_ID'] == -1:
                self._data_set[neighbour]['CLUSTER_ID'] = current_cluster_id

        return neighbourhood

    def _aggregate_data(self, gaze_points_list):
        """
        Aggregate data from @DATA of all arff objects in the input list into a
        new data set in form of a numpy array.

        :param gaze_points_list: gaze data to be clustered in form of list of arff objects.
        :return: data set to be clustered in form of a 6-column numpy array,
                 i.e. ['time','x','y','observer_id','CLUSTER_ID','visited_flag'],
                 ordered by 'time' column value.

        """
        data_set = []
        for i in range(len(gaze_points_list)):
            gaze_points_data = gaze_points_list[i]['data'][
                (gaze_points_list[i]['data']['EYE_MOVEMENT_TYPE'] == 'UNKNOWN')][['time', 'x', 'y', 'global_index']]
            gaze_points_data = ArffHelper.add_column_to_array(gaze_points_data, 'observer_id', 'NUMERIC',
                                                              gaze_points_list[i]['metadata']['observer_id'])
            gaze_points_data = ArffHelper.add_column_to_array(gaze_points_data, 'CLUSTER_ID', 'NUMERIC', -1)
            gaze_points_data = ArffHelper.add_column_to_array(gaze_points_data, 'visited_flag', 'NUMERIC', 0)
            if len(gaze_points_data) > 0:
                data_set.append(gaze_points_data)
        data_set = np.concatenate(data_set)
        data_set = np.sort(data_set, order='time')

        return data_set

    def _get_neighbourhood(self, current_point):
        """
        Get neighbourhood of current point in self._data_set (a 6-column numpy array as data set to be clustered)

        :param current_point: index of the current core point candidate.
        :return: index list of the neighbourhood of current point.

        """
        # cast to the appropriate type just in case
        start_index = np.searchsorted(self._timestamps,
                                      self._timestamps[current_point] - self._timestamps.dtype.type(self.time_slice),
                                      side='left')
        end_index = np.searchsorted(self._timestamps,
                                    self._timestamps[current_point] + self._timestamps.dtype.type(self.time_slice),
                                    side='right')

        distance = np.linalg.norm([self._data_set[start_index:end_index]['x'] - self._data_set[current_point]['x'],
                                   self._data_set[start_index:end_index]['y'] - self._data_set[current_point]['y']],
                                  axis=0)
        neighbourhood = (np.where(distance <= self.eps_px)[0] + start_index).tolist()

        return neighbourhood

    @abc.abstractmethod
    def _validate_neighbourhood(self, *args, **kwargs):
        """
        Should return a boolean value after neighbourhood validation. Returns True if the point with such neighbourhood
        is a core point (see DBSCAN method explanation for details).

        Abstract method - implemented in subclasses.

        """
        raise NotImplementedError("Implemented in subclass methods.")


class DBSCANWithMinPts(DBSCANWithTimeSlice):
    """
    DBSCAN with time slice that uses MinPts as neighbourhood validation method
    (validating that the number of other gaze points in the neighbourhood is at least @min_pts before declaring
    this a core point).

    This method is dependent on the frame rate of the gaze position recording, since the number of points in a
    fixed temporal slice will grow proportionally to gaze recording fps. If more independence from the fps is desired,
    use DBSCANWithMinObservers. Tha default value here was used on a dataset with 250 Hz tracker used.

    """
    def __init__(self, eps_deg=2.0, time_slice_microsec=40000, min_pts='num_observers'):
        """
        Initialize DBSCANWithMinPts object.
        :param eps_deg: Spatial Euclidean distance threshold that defines the neighbourhood in the XY-plane.
                        Given in degrees of visual field, a pixel value is assigned when the recordings' data
                        is provided.
        :param time_slice_microsec: Width of the time slice that defines the size of the neighbourhood on the time axis.
                                    Value is given in microseconds.
        :param min_pts: integer indicating the minimum number of points required to
                        form a "valid" neighbourhood (that of a core point).
                        Could also be a 'num_observers' string (default), in which case
                        the actual value is determined during the self._setup_internal_parameters() call
        """
        super(DBSCANWithMinPts, self).__init__(eps_deg=eps_deg, time_slice_microsec=time_slice_microsec)
        assert isinstance(min_pts, int) or min_pts == 'num_observers'
        self.min_pts_abs_value = None  # will initialize when the data is provided, in case it is `num_observers`

        self.min_pts = min_pts
        if type(self.min_pts) == int:
            self.min_pts_abs_value = self.min_pts

    def _setup_internal_parameters(self, gaze_points_list):
        """
        If min_pts was 'num_observers', set it accordingly here
        :param gaze_points_list: a list of arff objects (dictionary with fields such as 'data' and 'metadata')

        """
        super(DBSCANWithMinPts, self)._setup_internal_parameters(gaze_points_list)
        if self.min_pts == 'num_observers':
            self.min_pts_abs_value = len(gaze_points_list)

    def _validate_neighbourhood(self, neighbourhood):
        """
        Compare the size of @neighbourhood with @self.min_pts and return boolean value
        as result of validation. True if this is the neighbourhood of a core point, false otherwise.
        @self._data_set (a 6-column numpy array as data set to be clustered) is used to interpret
        the @neighbourhood list.

        :param neighbourhood: index list as neighbourhood to be validated.
        :return: boolean value.
                 True if @neighbourhood contains more than @self.min_pts points, False if not.

        """
        if len(neighbourhood) >= self.min_pts_abs_value:
            return True
        else:
            return False


class DBSCANWithMinObservers(DBSCANWithTimeSlice):
    """
    DBSCAN with time slice that uses MinObservers as neighbourhood validation method
    (validates that samples of at least @min_observers different observers are present in the neighbourhood before
    declaring this a core point).
    This method is more robust to varying frame rates of the recordings, since its outcome should not change if we
    switch from 250 fps to 500 fps, provided that @time_slice_microsec stays the same.

    """
    def __init__(self, eps_deg=2.0, time_slice_microsec=40000, min_observers=0.1):
        """
        Initialize DBSCANWithMinObservers object.

        :param eps_deg: Spatial Euclidean distance threshold that defines the neighbourhood in the XY-plane.
                        Given in degrees of visual field, a pixel value is assigned when the recordings' data
                        is provided.
        :param time_slice_microsec: Width of the time slice that defines the size of the neighbourhood on the time axis.
                                    Value is given in microseconds.
        :param min_observers: either float [0; 1] (indicating the share of all the
                              present observers) or int [2; +\inf) (indicating the
                              absolute threshold for observer count).
        """
        super(DBSCANWithMinObservers, self).__init__(eps_deg=eps_deg, time_slice_microsec=time_slice_microsec)

        self.min_observers_abs_value = None  # will initialize when the data is provided
        if (isinstance(min_observers, float) and 0. <= min_observers <= 1.) or \
                (isinstance(min_observers, int) and min_observers >= 2):
            self.min_observers = min_observers
            if isinstance(self.min_observers, int):
                self.min_observers_abs_value = self.min_observers
        else:
            raise ValueError(
                '@min_observers parameter should be either float [0; 1] '
                '(indicating the share of all the present observers)'
                ' or int [2; +\inf) (indicating the absolute threshold for observer count). Got {}'.
                format(min_observers))

    def _setup_internal_parameters(self, gaze_points_list):
        """
        If @self.min_min_observers was a float (i.e. share of observer count), set it's value for the provided
        data set here.

        """
        super(DBSCANWithMinObservers, self)._setup_internal_parameters(gaze_points_list)
        if type(self.min_observers) == float:
            self.min_observers_abs_value = round(self.min_observers * len(gaze_points_list))

    def _validate_neighbourhood(self, neighbourhood):
        """
        Compare the set of different observers in the @neighbourhood with @self.min_observers and return boolean value
        as result of validation. True if this is the neighbourhood of a core point, false otherwise.
        @self._data_set (a 6-column numpy array as data set to be clustered) is used to interpret
        the @neighbourhood list.

        :param neighbourhood: index list as neighbourhood to be validated.
        :return: boolean value.
                 True if @neighbourhood contains gaze points of at least @self.min_observers individual observers,
                 False if not.

        """
        number_of_users = self._get_number_of_unique_observers(neighbourhood)
        return number_of_users >= self.min_observers_abs_value

    def _get_number_of_unique_observers(self, neighbourhood):
        """
        Compute the number of different observers in the @neighbourhood.
        @self._data_set (a 6-column numpy array as data set to be clustered) is used to interpret
        the @neighbourhood list.

        :param neighbourhood: index list as neighbourhood to be validated.
        :return: the number of unique observers in the @neighbourhood

        """
        return len(set(self._data_set[neighbourhood]['observer_id']))

