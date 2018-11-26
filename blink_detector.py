import copy
import sys
import numpy as np


class BlinkDetector(object):
    """
    Detecting blinks by extending the 0-confidence intervals into nearby saccades. The maximal distance to saccade is
    an initialization parameter. When the observer performs a blink, the eye tracker usually first detects a saccade
    downwards, then looses the eye image, then detects a saccade upwards when the eye is opened.
    """
    def __init__(self, max_distance_to_saccade_microsec=25000, verbose=False):
        """
        Initialize BlinkDetector object.
        :param max_distance_to_saccade_microsec: threshold for distance from a definite blink to a nearby saccade,
                                                 which will be marked as blink as well.
        :param verbose: whether to output any progress and/or other detection-related information
        """
        self.MAXIMAL_DISTANCE_TO_SACCADE_MICROSEC = max_distance_to_saccade_microsec
        self.verbose = verbose

    def detect(self, gaze_points, inplace=False):
        """
        This method labels blinks in the provided gaze_points, which should be an arff object. We extend the
        0-confidence intervals by adding the nearest saccade (up to one from the left and up to one from the right)
        if it is no more than @self.MAXIMAL_DISTANCE_TO_SACCADE_MICROSEC away from the 0-confidence sample.


        :param gaze_points: gaze recording data, an arff object (i.e. a dictionary with 'data', 'metadata'
                            and etc. keys). If the array under the 'data' key has no 'confidence' column, the method
                            does nothing.
        :param inplace: whether to replace the data inside @gaze_points or create a new structure
        :return: gaze points with added labels BLINK
        """
        if not inplace:
            gaze_points = copy.deepcopy(gaze_points)

        if 'confidence' not in gaze_points['data'].dtype.names:
            return gaze_points

        # 0 and 1 array
        is_blink = (gaze_points['data']['confidence'] == 0).astype(int)
        # find blink onsets
        # fake not-blink sample before recording
        blink_diff = np.diff(np.hstack([[0], is_blink]))
        blink_onsets = np.nonzero(blink_diff == 1)[0]
        # find blink offsets
        # fake not-blink sample after recording
        blink_diff = np.diff(np.hstack([is_blink, [0]]))
        blink_offsets = np.nonzero(blink_diff == -1)[0]

        times = gaze_points['data']['time']
        assert len(blink_onsets) == len(blink_offsets)
        for onset, offset in zip(blink_onsets, blink_offsets):
            if self.verbose:
                print >> sys.stderr, "Found blink from {} to {}".format(
                    times[onset], times[offset]
                )
            # go back in time and look for a saccade
            onset_candidate = onset
            while onset_candidate >= 0 \
                    and times[onset] - times[onset_candidate] < self.MAXIMAL_DISTANCE_TO_SACCADE_MICROSEC:
                if gaze_points['data'][onset_candidate]['EYE_MOVEMENT_TYPE'] == 'SACCADE':
                    # Found a saccade! The blink will start at the start of this saccade
                    sacc_index = gaze_points['data'][onset_candidate]['SACC_INTERVAL_INDEX']
                    first_saccade_index = np.nonzero(
                        gaze_points['data']['SACC_INTERVAL_INDEX'] == sacc_index)[0][0]
                    onset = first_saccade_index
                    break
                # otherwise just continue the search backwards
                onset_candidate -= 1

            # go forward in time and look for a saccade
            offset_candidate = offset
            while offset_candidate < len(times) \
                    and times[offset_candidate] - times[offset] < self.MAXIMAL_DISTANCE_TO_SACCADE_MICROSEC:
                if gaze_points['data'][offset_candidate]['EYE_MOVEMENT_TYPE'] == 'SACCADE':
                    # Found a saccade! The blink will end at the end of this saccade
                    sacc_index = gaze_points['data'][offset_candidate]['SACC_INTERVAL_INDEX']
                    last_saccade_index = np.nonzero(
                        gaze_points['data']['SACC_INTERVAL_INDEX'] == sacc_index)[0][-1]
                    offset = last_saccade_index
                    break
                # otherwise just continue the search forwards
                offset_candidate += 1
            if self.verbose:
                print >> sys.stderr, "Extended it to {} {}".format(
                    times[onset], times[offset]
                )
            gaze_points['data'][onset:offset + 1]['EYE_MOVEMENT_TYPE'] = 'BLINK'
            # this is not a saccade anymore
            gaze_points['data'][onset:offset + 1]['SACC_INTERVAL_INDEX'] = -1
            # nor is it a normal sequence between saccades
            gaze_points['data'][onset:offset + 1]['INTERSACC_INTERVAL_INDEX'] = -1
        return gaze_points
