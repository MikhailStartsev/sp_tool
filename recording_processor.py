#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
import warnings

import data_loaders
from saccade_detector import SaccadeDetector
from blink_detector import BlinkDetector
from fixation_detector import FixationDetector
import util

EM_TYPE_ATTRIBUTE_NAME = 'EYE_MOVEMENT_TYPE'
EM_TYPE_ARFF_DATA_TYPE = ['UNKNOWN', 'FIX', 'SACCADE', 'SP', 'NOISE', 'BLINK', 'NOISE_CLUSTER', 'PSO']
EM_TYPE_DEFAULT_VALUE = 'UNKNOWN'


class RecordingProcessor:
    """
    The class is used for loading the recordings and pre-filtering them (i.e. detecting saccades and fixations).

    - Gaze data of one or several observers is loaded into arff objects.

    - Besides existing columns in the loaded arff objects, several extra columns are
      added into the @DATA section:
        - 'EYE_MOVEMENT_TYPE': a string chosen among 'UNKNOWN', 'FIX', 'SACCADE', 'BLINK', 'SP', 'NOISE'
          and 'NOISE_CLUSTER' (the latter is to indicate that a gaze sample has been labelled as noise by
          the clustering algorithm, and not by any of the earlier detectors)
        - 'SACC_INTERVAL_INDEX': an integer indicating data is in the n-th
           saccade interval (n >= 0, or -1 if not a saccade sample).
        - 'INTERSACC_INTERVAL_INDEX': an integer indicating data is in the
           n-th interval between two saccades (n >= 0, or -1 if not in any valid intersaccadic interval).

    - Some data points in the intersaccadic intervals are labeled as 'FIX' or 'NOISE', according to the fixation
      detection parameters.

    """

    def __init__(self, saccade_detector=None, blink_detector=None, fixation_detector=None):
        """
        Initialize the RecordingProcessor class

        :param saccade_detector: the initialized object of SaccadeDetector class; if None, default init is used
        :param blink_detector: the initialized object of BlinkDetector class; if None, default init is used
        :param fixation_detector: the initialized object of FixationDetector class; if None, default init is used
        """
        self._saccade_detector = saccade_detector if saccade_detector is not None else SaccadeDetector()
        self._blink_detector = blink_detector if blink_detector is not None else BlinkDetector()
        self._fixation_detector = fixation_detector if fixation_detector is not None else FixationDetector()

        # loaders for different formats/sources of data
        # should be all capital letters
        self._format_loaders = {
            'DSF': data_loaders.load_DSF_coord_as_arff_object,
            'ARFF': data_loaders.load_ARFF_as_arff_object,
            # This one is for arff files with already labelled eye movements, at least FIX and SACCADE types.
            # It can be used either to load results of a different algorithm, or preprocessed data with partially
            # labelled eye movements (i.e. your own saccade and fixation detector); is this type is chosen,
            # nor saccade/blink/fixation detectors will be involved.
            'LABELLED ARFF': data_loaders.load_ARFF_as_arff_object
        }

    def load_recording(self, fname, data_format=None,
                       labelled_eye_movement_column_arff=None, labelled_eye_movement_mapping_dict_arff=None,
                       suppress_warnings=False):
        """
        Load gaze data file @fname into arff object. This method calls on saccade, blink and fixation detectors.
        Also remembers the file name in 'metadata' section.

        :param fname: path to the file to load
        :param data_format: From which format to load the coordinates.
                            If None, will attempt to detect automatically by file extension.

                            Otherwise, can be one of the following:
                             - DSF, load from DSF .coord file
                             - ARFF, load .arff files


        The following parameters are relevant if you want to load a pre-labelled ARFF file with eye movement types being
        stored in a field other than EYE_MOVEMENT_TYPE with categorical values.

        CAUTION: in this case no saccades/blinks/fixations will be detected by this framework

        :param labelled_eye_movement_column_arff: the attribute that should be treated as an indication
                                                  of eye movement type, optional
        :param labelled_eye_movement_mapping_dict_arff: a dictionary that is used to convert values in column
                                                        @eye_movement_type_attribute to values in the following set:
                                                        ['UNKNOWN', 'FIX', 'SACCADE', 'SP', 'NOISE', 'BLINK',
                                                        'NOISE_CLUSTER'] (as defined by recording_processor.py)
        :param suppress_warnings: do not warn about the loaded data being assumed to have eye movement labels already

        :return: arff object with labelled 'SACCADE's, 'FIX's and 'BLINK's

        Example:
        recording = load_recording('test_data/YFK_welpen_20s_1.coord')
        """

        additional_args = {}
        if data_format is None:
            if fname.lower().endswith('.coord'):
                data_format = 'DSF'
            elif fname.lower().endswith('.arff'):
                if labelled_eye_movement_column_arff is not None or labelled_eye_movement_mapping_dict_arff is not None:
                    data_format = 'labelled ARFF'
                    additional_args['eye_movement_type_attribute'] = labelled_eye_movement_column_arff
                    additional_args['eye_movement_type_mapping_dict'] = labelled_eye_movement_mapping_dict_arff
                else:
                    data_format = 'ARFF'
            else:
                raise ValueError('The @data_format was not provided and could not be automatically detected. '
                                 'Please pass the appropriate @data_format (supported are {}) or convert your '
                                 'data to ARFF format with %@METADATA fields "width_px", "height_px", '
                                 '"width_mm", "height_mm" '
                                 'and "distance_mm". The attributes should include time, x and y columns.'.
                                 format(', '.join(list(self._format_loaders.keys()))))

        gaze_points = self._format_loaders[data_format.upper()](fname, **additional_args)
        gaze_points['metadata']['filename'] = fname
        util.add_eye_movement_attribute(gaze_points)

        if not data_format.startswith('labelled'):
            # mark saccades and at the same time label saccadic intervals and intersaccadic intervals
            # with respective IDs (important for subsequent fixation detection!)
            self._saccade_detector.detect(gaze_points, inplace=True)
            # mark blinks (extend 0-confidence intervals), remove IDs of saccadic and intersaccadic intervals for the
            # detected blink samples
            self._blink_detector.detect(gaze_points, inplace=True)
            # mark fixations (inside the previously detected intersaccadic intervals)
            self._fixation_detector.detect(gaze_points, inplace=True)
        elif not suppress_warnings:
            warnings.warn('The data format "{}" is selected, hence the steps of saccade/blink/fixation detection '
                          'are omitted! If this is not the desired behaviour, check the function help.'.
                          format(data_format))
        return gaze_points

    def load_multiple_recordings(self, fnames, data_format=None, validate_ppd=True,
                                 labelled_eye_movement_column_arff=None, labelled_eye_movement_mapping_dict_arff=None,
                                 verbose=False,
                                 suppress_warnings=False):
        """
        Load multiple gaze data files into a list of arff objects with saccade intervals labeled.

        New keyword 'observer_id' is added into @METADATA section of resulting ARFF objects, which is used to identify
        different observers.

        :param fnames: a list of paths to gaze data files.
        :param data_format: From which format to load the coordinates.
                            If None, will attempt to detect automatically.
                            Otherwise, can be one of the following:
                             - DSF, load from DSF .coord file
                             - ARFF, load ARFF data
        :param validate_ppd: whether to check that all the ppd values are the same
                             (should probably set to False if loading recordings with clips of different resolution
                             or viewing conditions at the same time; normally would load data for one clip at a time
                             through this method); setting it to True requires the presence of
                             'width_px', 'height_px', 'distance_mm', 'width_mm' and 'height_mm'
                             information in all of the loaded files!

        The following arguments are passed directly into RecordingProcessor load_recording() function.
        They are relevant if you want to load a pre-labelled ARFF file with eye movement types being
        stored in a field other than EYE_MOVEMENT_TYPE with categorical values.

        :param labelled_eye_movement_column_arff: the attribute that should be treated as an indication
                                                  of eye movement type, optional
        :param labelled_eye_movement_mapping_dict_arff: a dictionary that is used to convert values in column
                                                        @eye_movement_type_attribute to values in the following set:
                                                        ['UNKNOWN', 'FIX', 'SACCADE', 'SP', 'NOISE', 'BLINK',
                                                        'NOISE_CLUSTER'] (as defined by recording_processor.py)

        If you are passing not-None values for these arguments, no ppd validation will be performed
        (@validate_ppd=False is assumed), since it requires metadata keys in ARFF files, which
        are not actually needed for the most useful case of this case: loading data for immediate
        evaluation. If you want to validate the ppd nevertheless, call
        >> RecordingProcessor.validate_ppd_of_multiple_recordings(result)
        with @result being the result of this function

        :param verbose: whether to output progress information
        :param suppress_warnings: do not warn about not performing the PPD-consistency check
        :return: list of arff objects corresponding to the file names in @fnames

        Example:
        load_multiple_recordings(['test_data/YFK_breite_strasse_1_20s_1.coord',
                                  'test_data/AAF_breite_strasse_1_20s_1.coord'])
        """
        if labelled_eye_movement_column_arff is not None or labelled_eye_movement_mapping_dict_arff is not None:
            validate_ppd = False
            if not suppress_warnings:
                warnings.warn('The passed arguments correspond to labelled ARFF format, pixel-per-degree value '
                              'equality for all recordings validation step is omitted in this case. If this is not '
                              'the desired behaviour, check function help.')
        res = []
        observer_id = 0
        if verbose:
            print('Loading {} files:'.format(len(fnames)), file=sys.stderr)
        for i, fname in enumerate(fnames):
            gaze_points = self.load_recording(fname,
                                              data_format=data_format,
                                              labelled_eye_movement_column_arff=
                                              labelled_eye_movement_column_arff,
                                              labelled_eye_movement_mapping_dict_arff=
                                              labelled_eye_movement_mapping_dict_arff,
                                              suppress_warnings=suppress_warnings)
            # extract trail id, add it to meta
            gaze_points['metadata']['observer_id'] = observer_id
            observer_id += 1
            # store into res
            res.append(gaze_points)
            if verbose:
                util.update_progress((i + 1, len(fnames)))
        print(file=sys.stderr)
        if validate_ppd:
            RecordingProcessor.validate_ppd_of_multiple_recordings(res)
        return res

    @staticmethod
    def validate_ppd_of_multiple_recordings(gaze_points_list, relative_tolerance=0.1):
        """
        Compare the PPD (pixel-per-degree) values when loading multiple recordings to verify that
        all of the PPD values are identical.

        :param gaze_points_list: list of arff objects.
        :param relative_tolerance: tolerate some deviation of PPD values, as long as it is no more than
                                   (@relative_tolerance * mean PPD value).
        :return: PPD value if unique (or deviations below tolerance; then mean PPD).

        """
        ppds = []
        for i in range(len(gaze_points_list)):
            one_value = util.calculate_ppd(gaze_points_list[i])
            ppds.append(round(one_value, 2))  # round to 2 decimals to avoid machine precision issues
        if len(ppds) == 0:
            raise ValueError('Empty list of recordings provided')
        if len(set(ppds)) == 1:
            return ppds[0]
        else:
            mean_ppd = sum(ppds) / float(len(ppds))
            deviation = max(max(ppds) - mean_ppd, mean_ppd - min(ppds)) / mean_ppd
            assert deviation > 0
            if deviation > relative_tolerance:
                raise ValueError('PPD values are different (relative tolerance of {} was exceeded) among provided '
                                 'recordings: {}'.format(relative_tolerance, ppds))
            else:
                return mean_ppd
