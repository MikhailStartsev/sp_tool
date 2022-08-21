import copy
import math

import numpy as np

import util
from arff_helper import ArffHelper


class FixationDetector(object):
    def __init__(self,
                 prefiltering_interval_spread_threshold_degrees=2.0,
                 min_sp_duration_microsec=50000,
                 sliding_window_width_microsec=50000,
                 normalization_sliding_window_size_samples=5,
                 speed_threshold_degrees_per_sec=2.0,
                 intersaccadic_interval_duration_threshold_microsec=150000,
                 sliding_window_criterion='speed'):
        """
        Initialize the FixationDetector class object

        :param prefiltering_interval_spread_threshold_degrees: all the intersaccadic intervals with a bounding box
                                                               smaller less than this (on both sides) will be deemed
                                                               fixations

        Fixation removal (based on Ferrera paper)
        :param min_sp_duration_microsec: non-fixation episodes shorter than this threshold will be marked as NOISE
        :param normalization_sliding_window_size_samples: to smoothen the gaze movement, we normalize the data with
                                                          moving average of this many samples
        :param sliding_window_width_microsec: afterwards we look at the data with a sliding window of this width
        :param speed_threshold_degrees_per_sec: if magnitude of the average speed within the window is below this
                                                threshold, the samples within the window are labelled as FIXATION
        :param intersaccadic_interval_duration_threshold_microsec: if the intersaccadic interval is shorter than this
                                                                   threshold, we do not apply the sliding window
                                                                   processing from above (the interval is too short).
                                                                   Instead, we label the samples in this interval as
                                                                   UNKNOWN (since the spread on this short interval
                                                                   exceeds
                                                                   @prefiltering_interval_spread_threshold_degrees,
                                                                   it is too fast to be a fixation)
        :param sliding_window_criterion: should be either 'spread' or 'speed' ('speed' by default).
                                         This defines how to apply the @self.SPEED_THRESHOLD_DEGREES_PER_SEC threshold.

                                         If 'speed', we check the displacement of the last sample of the sliding window
                                         relative to the first sample of the window, and compute average speed based on
                                         this displacement and the time elapsed. Average speed is compared to
                                         @self.SPEED_THRESHOLD_DEGREES_PER_SEC.

                                         If 'spread', we compute the bounding box for samples within the current window
                                         and compare its size to
                                         @self.SPEED_THRESHOLD_DEGREES_PER_SEC * <time elapsed>.
        """
        # prefiltering parameters, applied per intersaccadic interval
        self.PREFILTERING_INTERVAL_SPREAD_THRESHOLD_DEGREES = prefiltering_interval_spread_threshold_degrees

        # fixation removal parameters
        self.MINIMAL_SP_DURATION_MICROSEC = min_sp_duration_microsec
        self.SLIDING_WINDOW_WIDTH_MICROSEC = sliding_window_width_microsec
        self.NORMALIZATION_SLIDING_WINDOW_SIZE_SAMPLES = normalization_sliding_window_size_samples
        self.SPEED_THRESHOLD_DEGREES_PER_SEC = speed_threshold_degrees_per_sec
        self.INTERSACCADIC_INTERVAL_DURATION_THRESHOLD_MICROSEC = intersaccadic_interval_duration_threshold_microsec

        assert sliding_window_criterion in {'speed', 'spread'}
        self.SLIDING_WINDOW_CRITERION = sliding_window_criterion

    def detect(self, gaze_points, inplace=False):
        """
        Identify and label fixation intervals as 'FIX' and some others as 'NOISE'.

        Fixation identification includes the following steps:
        - First, all inter-saccadic intervals with a dispersion of less than
          a certain spread threshold (@self.PREFILTERING_INTERVAL_SPREAD_THRESHOLD_DEGREES) are marked as fixations.
        - Then, a temporal window (@self.SLIDING_WINDOW_WIDTH_MICROSEC ms) is shifted across the
          remaining data and a non-fixation onset (offset) is marked every
          time speed rises above (fell below) threshold (@self.SPEED_THRESHOLD_DEGREES_PER_SEC).
        - There are two ways for speed calculation: spread and speed.
            -'speed': speed from start point to end point is larger than
                      threshold.
            -'spread': maximum moving speed of either x or y is larger than
                       threshold.
          Data with speed below threshold are labeled as 'FIX'.
        - Finally, non-fixation episodes longer than @self.MINIMAL_SP_DURATION_MICROSEC are kept as 'UNKNOWN',
          the shorter ones are labeled as 'NOISE' (these are fairly dynamic episodes that however should not be SP).

        :param gaze_points: arff object with saccades detected (and intersaccadic intervals labelled)
        :param inplace: whether to replace the data inside @gaze_points or create a new structure
        :return: arff object with data labeled as 'FIX' and 'NOISE'. Some 'UNKNOWN' labels are kept for the next stage.

        """
        if not inplace:
            gaze_points = copy.deepcopy(gaze_points)
        # add a global index column (to keep track of where we are even if working within an intersaccadic interval)
        gaze_points = ArffHelper.add_column(gaze_points, name='global_index', dtype='INTEGER', default_value=-1)
        gaze_points['data']['global_index'] = np.arange(gaze_points['data'].shape[0])

        # I. First step of fixation removal: rough prefiltering
        #
        # Convert constants to pixels per second
        ppd = util.calculate_ppd(gaze_points)
        speed_thd = ppd * self.SPEED_THRESHOLD_DEGREES_PER_SEC
        prefiltering_spread_thd = ppd * self.PREFILTERING_INTERVAL_SPREAD_THRESHOLD_DEGREES

        # record intersaccadic interval indices of those intervals that are not labelled as FIX by the prefiltering
        unknown_interval_index = []
        unknown_interval_masks = []
        for i in range(max(gaze_points['data']['INTERSACC_INTERVAL_INDEX']) + 1):
            mask = gaze_points['data']['INTERSACC_INTERVAL_INDEX'] == i
            intersacc_interval = gaze_points['data'][mask]
            if len(intersacc_interval) == 0:
                continue

            dispersion = [max(intersacc_interval['x']) - min(intersacc_interval['x']),
                          max(intersacc_interval['y']) - min(intersacc_interval['y'])]

            if any(thd >= prefiltering_spread_thd for thd in dispersion):
                unknown_interval_index.append(i)  # keep unknown
                unknown_interval_masks.append(mask.copy())  # cache the indexing
            else:
                gaze_points['data']['EYE_MOVEMENT_TYPE'][mask] = 'FIX'

        # II. Second step of fixation removal: finer prefiltering
        #

        for i, interval_mask in zip(unknown_interval_index, unknown_interval_masks):
            # We record the borders of the non-FIX episodes to validate their duration. If the non-FIX episode is very
            # short, we mark it as NOISE (not enough duration for a candidate for smooth pursuit)
            onset_timestamp = None
            onset_index = None

            intersacc_interval = gaze_points['data'][interval_mask]
            intersacc_interval = util.get_xy_moving_average(intersacc_interval,
                                                            self.NORMALIZATION_SLIDING_WINDOW_SIZE_SAMPLES,
                                                            inplace=False)

            # for intervals shorter than @self.INTERSACCADIC_INTERVAL_DURATION_THRESHOLD_MICROSEC:
            # cannot do further filtering. The label remains 'UNKNOWN'
            if intersacc_interval['time'][-1] - intersacc_interval['time'][0] < \
                    self.INTERSACCADIC_INTERVAL_DURATION_THRESHOLD_MICROSEC:
                continue

            # for intervals that longer than self.SLIDING_WINDOW_WIDTH_MICROSEC: do further pre-filtering.
            # Label data as 'FIX' or 'NOISE', or keep 'UNKNOWN'
            else:
                # window is shifted by 1 sample every time
                for index, item in enumerate(intersacc_interval):
                    x_start = item['x']
                    y_start = item['y']
                    shift_window_interval = intersacc_interval[
                        (intersacc_interval['time'] >= item['time']) *
                        (intersacc_interval['time'] <= item['time'] + self.SLIDING_WINDOW_WIDTH_MICROSEC)
                    ]

                    # if distance between current data and the end of interval is shorter than
                    # self.SLIDING_WINDOW_WIDTH_MICROSEC (i.e. if the end of the window matches the end of the
                    # intersaccadic interval), we keep the previous label if it was FIX, otherwise keep UNKNOWN
                    if shift_window_interval['time'][-1] == intersacc_interval['time'][-1]:
                        if intersacc_interval['EYE_MOVEMENT_TYPE'][index - 1] == 'FIX':
                            gaze_points['data']['EYE_MOVEMENT_TYPE'][
                                (gaze_points['data']['time'] == item['time'])] = 'FIX'

                            # we do not keep track of the non-fixation interval anymore since it will be all fixation
                            # until the end of the intersaccadic interval
                            onset_timestamp = None
                            onset_index = None
                        else:
                            # new non-fixation interval is starting
                            onset_timestamp = item['time']
                            onset_index = item['global_index']

                    # if distance between current data and the end of interval is larger than window size, continue
                    # with the process
                    else:
                        # get window duration in seconds
                        period = (shift_window_interval['time'][-1] - shift_window_interval['time'][0]) * 1e-6

                        # is the fixation criterion satisfied?
                        fixation_flag = True
                        if self.SLIDING_WINDOW_CRITERION == 'speed':
                            # if the current speed is larger than speed threshold --
                            # mark as onset(UNKNOWN, NOISE). else -- mark as offset(FIX)
                            x_end = shift_window_interval['x'][-1]
                            y_end = shift_window_interval['y'][-1]

                            if math.sqrt((x_start - x_end) ** 2 + (y_start - y_end) ** 2) >= speed_thd * period:
                                # will not be a fixation
                                fixation_flag = False
                        else:  # spread
                            # if either x_max - x_min or y_max - y_min is larger than speed threshold * time --
                            # mark as onset. else -- mark as offset
                            x_max = max(shift_window_interval['x'])
                            x_min = min(shift_window_interval['x'])
                            y_max = max(shift_window_interval['y'])
                            y_min = min(shift_window_interval['y'])

                            if max(x_max - x_min, y_max - y_min) >= speed_thd * period:
                                # will not be a fixation
                                fixation_flag = False

                        if fixation_flag:
                            gaze_points['data']['EYE_MOVEMENT_TYPE'][item['global_index']] = 'FIX'

                        # either a fixation start or the whole interval end
                        if fixation_flag or index == len(intersacc_interval) - 1:
                            # if we had a non-fixation interval going on before, check it's duration
                            if onset_index is not None:
                                # onset episode larger than 50ms: UNKNOWN. else: NOISE
                                if item['time'] - onset_timestamp < self.MINIMAL_SP_DURATION_MICROSEC:
                                    offset_timestamp = item['time'] - 1
                                    offset_index = item['global_index'] - 1
                                    # if this is not the beginning of fixation,
                                    # the last item also should be labelled as NOISE
                                    if not fixation_flag:
                                        offset_timestamp += 1
                                        offset_index += 1

                                    gaze_points['data'][onset_index:(offset_index + 1)]['EYE_MOVEMENT_TYPE'] = 'NOISE'

                                # episode is finished
                                onset_timestamp = None
                                onset_index = None
                        else:
                            # if new non-fixation interval started
                            if onset_timestamp is None:
                                onset_timestamp = item['time']
                                onset_index = item['global_index']
                            # otherwise it just continues, don't have to do anything
        # can now remove the global_index column
        gaze_points = ArffHelper.remove_column(gaze_points, 'global_index')
        return gaze_points
