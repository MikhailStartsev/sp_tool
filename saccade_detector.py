import sys
import copy

import numpy as np

import util
from arff_helper import ArffHelper


class SaccadeDetector(object):
    """
    Detecting saccades as in data_source_framework (DSF).
    """

    def __init__(self,
                 tolerance=0.0,  # 0.1,
                 threshold_onset_fast_degree_per_sec=137.5,
                 threshold_onset_slow_degree_per_sec=17.1875,
                 threshold_offset_degree_per_sec=17.1875,
                 max_speed_degree_per_sec=1031.25,
                 min_duration_microsec=15000,
                 max_duration_microsec=160000,
                 velocity_integral_interval_microsec=4000,
                 verbose=False
                 ):
        """

        :param tolerance: The relative size of the area outside the screen that is still considered to be legal
        :param threshold_onset_fast_degree_per_sec: Threshold for initialization of saccade detection.
                                                    This and the following thresholds are in deg/s.
        :param threshold_onset_slow_degree_per_sec: Once velocity is above :param thresholdOnsetFast, the first sample
                                                    where velocity was > thresholdOnsetSlow is marked as saccade onset
        :param threshold_offset_degree_per_sec: Maximum velocity that is regarded as end of saccade
        :param max_speed_degree_per_sec: Maximum speed of saccadic eye movements. Velocities exceeding this value are
                                         treated as impulse noise.
        :param min_duration_microsec: Saccades of a duration less than this (in microseconds) are treated as impulse
                                      noise.
        :param max_duration_microsec: Max duration that the saccade status is held upright. During a glitch,
                                      an offset might not be detected, so eventually the saccade flag is switched off
                                      anyway. This value is given in microseconds. The current default of 160000us
                                      roughly equals an amplitude of >50 degrees.
        :param velocity_integral_interval_microsec: Interval over which to integrate velocity computation.

                                                    For very high sampling rates (i.e., SMI's 1250 Hz), calculating
                                                    sample-to-sample velocity is not feasible.

                                                    The current default of 2000us means that for 240 Hz (SMI Hi-Speed)
                                                    or 250 Hz (EyeLink) recordings, only one sample is taken
                                                    into account as before; for 1250 Hz recordings, though, we compute
                                                    velocity on three samples.
        :param verbose: whether to output any progress and/or other detection-related information

        """
        self.TOLERANCE = tolerance
        self.THRESHOLD_ONSET_FAST_DEGREE_PER_SEC = threshold_onset_fast_degree_per_sec
        self.THRESHOLD_ONSET_SLOW_DEGREE_PER_SEC = threshold_onset_slow_degree_per_sec
        self.THRESHOLD_OFFSET_DEGREE_PER_SEC = threshold_offset_degree_per_sec
        self.MAX_SPEED_DEGREE_PER_SEC = max_speed_degree_per_sec
        self.MIN_DURATION_MICROSEC = min_duration_microsec
        self.MAX_DURATION_MICROSEC = max_duration_microsec
        self.VELOCITY_INTEGRAL_INTERVAL_MICROSEC = velocity_integral_interval_microsec

        self.verbose = verbose

    def detect(self, gaze_points, inplace=False):
        """
        This method labels saccades (also noise) in the provided gaze_points, which should be an arff object
        :param gaze_points: gaze recording data, an arff object (i.e. a dictionary with 'data', 'metadata'
                            and etc. keys)
        :param inplace: whether to replace the data inside @gaze_points or create a new structure
        :return: gaze points with added labels SACCADE, NOISE
        """
        if not inplace:
            gaze_points = copy.deepcopy(gaze_points)

        # also keep track of saccadic and intersaccadic intervals
        detected_saccades_count = 0
        if 'SACC_INTERVAL_INDEX' not in gaze_points['data'].dtype.names:
            ArffHelper.add_column(gaze_points, 'SACC_INTERVAL_INDEX', 'INTEGER', -1)

        # a virtual saccade that finished before the recording for uniform processing
        last_saccade_end = -1
        intersaccadic_intervals_count = 0
        if 'INTERSACC_INTERVAL_INDEX' not in gaze_points['data'].dtype.names:
            ArffHelper.add_column(gaze_points, 'INTERSACC_INTERVAL_INDEX', 'INTEGER', -1)

        # verify that the timestamps are sorted!
        times = gaze_points['data']['time']
        assert all(times[i] <= times[i + 1] for i in xrange(len(times) - 1)), \
            'Timestamps are not sorted in {}'.format(gaze_points['metadata']['filename'])
        # -1 so that the exact value ends up on the right of the searched timestamp
        searchable_timestamps = times - self.VELOCITY_INTEGRAL_INTERVAL_MICROSEC - 1
        # find the indices of the first
        prev_indices = np.searchsorted(times, searchable_timestamps, side='right')
        cur_indices = np.arange(len(prev_indices))
        # if the index after search points towards this very data point, take the previous one
        prev_indices[prev_indices == cur_indices] -= 1
        # except for the very first sample
        prev_indices[0] = 0

        # computing velocities
        x_shifts = gaze_points['data']['x'][cur_indices] - gaze_points['data']['x'][prev_indices]
        y_shifts = gaze_points['data']['y'][cur_indices] - gaze_points['data']['y'][prev_indices]
        shifts = np.linalg.norm(np.vstack([x_shifts, y_shifts]), axis=0)
        time_shifts = gaze_points['data']['time'][cur_indices] - gaze_points['data']['time'][prev_indices]
        # keep it above 0, the shifts are 0 there anyway
        time_shifts[time_shifts == 0] += 1

        velocities = shifts / time_shifts  # pixels per microsecond
        ppd = util.calculate_ppd(gaze_points)
        velocities /= ppd  # degree per microsecond
        velocities *= 1e6  # degree per second

        # How many samples back is it reasonable to go?
        time_step = np.diff(times).mean()
        # a big margin of error, 10 times as many samples as would normally need
        extra_samples_count = int(np.round((self.MAX_DURATION_MICROSEC * 10) / time_step))
        # Glitch detection: glitches are defined by one of several features.
        #
        # (1) Coordinates far outside the calibrated region (what constitutes far is defined
        # by the tolerance parameter) are assumed to be erroneous.
        is_glitch = np.zeros(gaze_points['data'].shape[0], dtype=np.bool)
        is_glitch[gaze_points['data']['x'] < -gaze_points['metadata']['width_px'] * self.TOLERANCE] = True
        is_glitch[gaze_points['data']['y'] < -gaze_points['metadata']['height_px'] * self.TOLERANCE] = True
        is_glitch[gaze_points['data']['x'] > gaze_points['metadata']['width_px'] * (1 + self.TOLERANCE)] = True
        is_glitch[gaze_points['data']['y'] > gaze_points['metadata']['height_px'] * (1 + self.TOLERANCE)] = True

        # (2) If the @gaze_points supports the estimate of a confidence
        # measure for samples, a confidence lower than 0.1 also indicates
        # a glitch here.
        if 'confidence' in gaze_points['data'].dtype.names:
            is_glitch[gaze_points['data']['confidence'] < 0.1] = True

        # (3) Finally, velocities that exceed \a maxSpeed (default currently
        # set to ~1000 degrees/s) are regarded as glitches as well and labelled as noise
        is_glitch[velocities > self.MAX_SPEED_DEGREE_PER_SEC] = True
        gaze_points['data']['EYE_MOVEMENT_TYPE'][velocities > self.MAX_SPEED_DEGREE_PER_SEC] = 'NOISE'

        # Remember first sample after glitch:
        # to prevent saccade detection at the first non-glitch sample
        # that follows, saccade detection is inhibited for that first sample.
        post_glitch = np.diff(is_glitch.astype(int)) == -1
        post_glitch = np.hstack(([False], post_glitch))
        # Remember last sample before glitch:
        # since we normally would suspend the other criteria (incl. speed) if we are inside glitch, we try to avoid
        # border effects in both next-after and last-before glitch samples
        pre_glitch = np.diff(is_glitch.astype(int)) == 1
        pre_glitch = np.hstack((pre_glitch, [False]))
        all_glitch = is_glitch + post_glitch + pre_glitch
        # we will assign glitch samples' labels to NOISE after the saccades have been detected

        # recompute speeds for post-glitch samples
        pre_glitch_indices = np.nonzero(pre_glitch)[0]
        for i in np.nonzero(post_glitch)[0]:
            # find the corresponding start of the glitch
            corresponding_pre_glitch = np.searchsorted(pre_glitch_indices, i) - 1
            if corresponding_pre_glitch < 0:
                # no correspondence found, it's the glitch from the beginning of recording ==> set velocity to 0
                velocities[i] = 0
            else:
                # found a completed glitch
                velocities[i] = np.linalg.norm([
                    gaze_points['data']['x'][i] - gaze_points['data']['x'][corresponding_pre_glitch],
                    gaze_points['data']['y'][i] - gaze_points['data']['y'][corresponding_pre_glitch]
                ]) / (times[i] - times[corresponding_pre_glitch])  # pixels per microsecond
                velocities[i] /= ppd  # degrees per microsecond
                velocities[i] *= 1e6  # degrees per second

        # Looking for saccade seed points
        # saccade seed point should
        # (1) exceed the fast threshold
        # (2) be biologically plausible
        # (3) not be inside a glitch
        saccade_seeds = (velocities > self.THRESHOLD_ONSET_FAST_DEGREE_PER_SEC) * \
                        (velocities < self.MAX_SPEED_DEGREE_PER_SEC) * \
                        (1 - all_glitch)
        saccade_seed_indices = np.nonzero(saccade_seeds)[0]
        for potential_seed_index in saccade_seed_indices:
            if gaze_points['data']['EYE_MOVEMENT_TYPE'][potential_seed_index] != 'UNKNOWN':
                # already labelled this before, ex. as a saccade that started from another seed point
                continue
            if self.verbose == 2:
                print >> sys.stderr, 'potential seed index', potential_seed_index
            # Looking for onset:
            # (1) should be above slow threshold speed
            # (2) should not be a glitch
            # (3) does not yet have a label
            onset_candidates_check = (velocities[max(0, potential_seed_index - extra_samples_count):potential_seed_index] >=
                                      self.THRESHOLD_ONSET_SLOW_DEGREE_PER_SEC) * \
                                     (1 - is_glitch[max(0, potential_seed_index - extra_samples_count):potential_seed_index]) * \
                                     (gaze_points['data']['EYE_MOVEMENT_TYPE'][
                                          max(0, potential_seed_index - extra_samples_count):potential_seed_index
                                      ] == 'UNKNOWN')

            # find the last zero (the next sample after it is the beginning of the last uninterrupted 1-sequence,
            # i.e. the saccade onset
            try:
                last_zero_index = np.nonzero(1 - onset_candidates_check)[0][-1]
            except IndexError:
                # not found
                continue
            saccade_onset_index = last_zero_index + 1 + max(0, potential_seed_index - extra_samples_count)  # shift accordingly
            # also this should not be the glitch or post/pre-glitch sample
            while all_glitch[saccade_onset_index]:
                saccade_onset_index += 1

            # looking for offset
            # (1) should be above offset speed threshold
            # (2) should not exceed biologically plausible duration threshold
            # (3) should not yet have a label (i.e. not NOISE labelled above)
            offset_candidates_check = (velocities[potential_seed_index:potential_seed_index + extra_samples_count] >=
                                       self.THRESHOLD_OFFSET_DEGREE_PER_SEC) * \
                                      (times[potential_seed_index:potential_seed_index + extra_samples_count] -
                                       times[saccade_onset_index] <= self.MAX_DURATION_MICROSEC)
            # we ignore the criterion around the glitch
            offset_candidates_check += is_glitch[potential_seed_index:potential_seed_index + extra_samples_count]
            offset_candidates_check += post_glitch[potential_seed_index:potential_seed_index + extra_samples_count]

            # but there should not yet be a label present, i.e. it's not the NOISE labelled above
            offset_candidates_check *= (gaze_points['data']['EYE_MOVEMENT_TYPE'][
                                            potential_seed_index:potential_seed_index + extra_samples_count
                                        ] == 'UNKNOWN')

            # find the first zero (this is the first sample with speed below the threshold, i.e. the saccade offset
            try:
                saccade_offset_index = np.nonzero(1 - offset_candidates_check)[0][0]
            except IndexError:
                # no offset found
                continue
            # the index was starting at potential_seed_index
            saccade_offset_index += potential_seed_index

            # if we are finished inside the glitch, we have reached a biological limit of some sorts ==> discard
            if is_glitch[saccade_offset_index]:
                continue

            if self.verbose == 2:
                print >> sys.stderr, 'Found onset/offset indices', saccade_onset_index, saccade_offset_index

            # now validate the saccade parameters
            # (1) it spans at least the minimal necessary interval
            saccade_time = times[saccade_offset_index] - times[saccade_onset_index]
            if saccade_time < self.MIN_DURATION_MICROSEC:
                # If the resulting saccade is shorter than
                # a minDuration, we assume that we have only encountered
                # some noise impulse and discard this saccade.
                gaze_points['data']['EYE_MOVEMENT_TYPE'][saccade_onset_index:saccade_offset_index + 1] = 'NOISE'

                if self.verbose == 2:
                    print >> sys.stderr, 'Discarding due to low duration: needed {}, had {}'.\
                        format(self.MIN_DURATION_MICROSEC, saccade_time)
                continue

            # (2) mean velocity is not below the slow onset threshold
            saccade_displacement = np.linalg.norm([
                gaze_points['data']['x'][saccade_offset_index] - gaze_points['data']['x'][saccade_onset_index],
                gaze_points['data']['y'][saccade_offset_index] - gaze_points['data']['y'][saccade_onset_index],
            ])
            mean_speed = saccade_displacement / saccade_time  # pixels per microsecond
            mean_speed /= ppd  # degrees per microsecond
            mean_speed *= 1e6  # degrees per second
            if mean_speed < self.THRESHOLD_ONSET_SLOW_DEGREE_PER_SEC:
                # Saccades where the average velocity drops below the offset threshold
                # are also discarded (those are often due to some high-velocity samples
                # going in one direction, then jumping back - which is unbiological).
                if self.verbose == 2:
                    print >> sys.stderr, 'Discarding due to low average speed: needed {}, had {}'.format(
                        self.THRESHOLD_ONSET_SLOW_DEGREE_PER_SEC, mean_speed)
                continue

            # If all is okay, we detected a whole saccade
            gaze_points['data']['EYE_MOVEMENT_TYPE'][saccade_onset_index:saccade_offset_index + 1] = 'SACCADE'
            # write the saccade index into the appropriate field and update the global count
            gaze_points['data']['SACC_INTERVAL_INDEX'][saccade_onset_index:saccade_offset_index + 1] = \
                detected_saccades_count
            detected_saccades_count += 1
            # from the end of last saccade till the beginning of this one, put appropriate intersaccadic interval index
            # also update the global count of intersaccadic intervals
            gaze_points['data']['INTERSACC_INTERVAL_INDEX'][last_saccade_end + 1:saccade_onset_index] = \
                intersaccadic_intervals_count
            intersaccadic_intervals_count += 1
            last_saccade_end = saccade_offset_index

            if self.verbose:
                print >> sys.stderr, '{0} {1:0.1f} {2:0.1f} {3} {4:0.1f} {5:0.1f}'.format(
                    gaze_points['data'][saccade_onset_index]['time'],
                    gaze_points['data'][saccade_onset_index]['x'],
                    gaze_points['data'][saccade_onset_index]['y'],
                    gaze_points['data'][saccade_offset_index]['time'],
                    gaze_points['data'][saccade_offset_index]['x'],
                    gaze_points['data'][saccade_offset_index]['y'],
                )
        # final intersaccadic interval, if there is one
        gaze_points['data']['INTERSACC_INTERVAL_INDEX'][last_saccade_end + 1:] = \
            intersaccadic_intervals_count
        intersaccadic_intervals_count += 1

        # Override erroneous samples' labels
        gaze_points['data']['EYE_MOVEMENT_TYPE'][is_glitch] = 'NOISE'
        return gaze_points



