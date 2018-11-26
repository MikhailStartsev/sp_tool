from collections import OrderedDict
import os
import numpy as np

from sp_tool.arff_helper import ArffHelper
from sp_tool import recording_processor


# Example usage (from sp_tool source directory)
# $> python
# >> from examples import convert_eyelink_asc_to_arff
# >> convert_eyelink_asc_to_arff.convert(fname='test_data/eyelink_data_example.asc',
#                                        corresponding_video_name='eyelink_example_video',
#                                        video_width_mm=440, video_height_mm=340,
#                                        distance_observer_to_screen_mm=800,
#                                        video_width_px=1280, video_height_px=720,
#                                        corneal_reflection_mode=True, binocular_flag=True,
#                                        extract_events_for_eye='R')


def convert(fname, corresponding_video_name,
            video_width_mm, video_height_mm, distance_observer_to_screen_mm, video_width_px, video_height_px,
            corneal_reflection_mode=True, binocular_flag=True, velocity_flag=False, resolution_flag=False,
            extract_events_for_eye=None,
            out_folder=None,
            restrict_time_low=None, restrict_time_high=None,
            gaze_time_modifier_microsec=lambda t: t * 1000, gaze_x_modifier=lambda x: x, gaze_y_modifier=lambda y: y
            ):
    """
    Convert (part of) EyeLink ASCII (.asc) text file
    into an ARFF file in respective folder (a subfolder in @out_folder is created for the @corresponding_video_name).
    The @fname basename is preserved, so don't hesitate to execute multiple convert() calls with the same @out_folder.
    By default, assuming binocular corneal reflection mode.

    The attributes of @time_column_name, @gaze_x_column_name and @gaze_y_column_name are placed on the first 3 spots
    in the resulting arff for readability.


    :param fname: *.asc file to be converted
    :param corresponding_video_name: video name (will create a subdirectory in @out_folder with this name)

    :param corneal_reflection_mode: whether corneal reflection mode (CR) was used during recording;
                                    if True, assume the last "column" consisting of characters '.', 'I', 'C' and 'R'
                                    (this column is largely ignored, confidence values come from valid or invalid
                                    coordinates for all eyes).
    :param binocular_flag: boolean, whether this is a binocular recording file; True by default
    :param velocity_flag: boolean, whether velocity is included in the file format; False by default
    :param resolution_flag: boolean, whether resolution data is include in file format; False by default

    :param extract_events_for_eye: can be None (= do not extract events), 'L' (= for left eye) or 'R' (= for right eye)

    :param video_width_mm: width of the video surface in mm
    :param video_height_mm: height of the video surface in mm
    :param distance_observer_to_screen_mm: distance from observer's eyes to the video surface in mm
    :param video_width_px: width of the video surface in pixels
    :param video_height_px: height of the video surface in mm

    :param out_folder: output folder, by default - the same as where @fname is located

    :param restrict_time_low: lower border for time (if we want to extract just part of the samples), in the same
                              units as in the file @fname!
    :param restrict_time_high: upper border for time (if we want to extract just part of the samples), in the same
                               units as in the file @fname!

    Main fields' modifiers. You can do this as post-processing as well, though (might be more intuitive).
    :param gaze_time_modifier_microsec: a function (one value in, one value out) to modify the timestamp of a gaze
                                        sample to produce microseconds; no change by default;
                                        NB: by default, assuming time in milliseconds, i.e. default
                                        @gaze_time_modifier_microsec multiplies the time column by 1000 !

    :param gaze_x_modifier: a function (one value in, one value out) to modify the X coordinate of gaze
                            (ex. if the video was scaled/shifted relative to tracked coordinates and you
                            want to immediately recover the in-video coordinates); no change by default
    :param gaze_y_modifier: a function (one value in, one value out) to modify the Y coordinate of gaze
                            (ex. if the video was scaled/shifted relative to tracked coordinates and you
                            want to immediately recover the in-video coordinates)
    :return: the output file name
    """
    assert extract_events_for_eye in {None, 'L', 'R'}
    convert.EVENT_STRINGS = ['SFIX', 'EFIX', 'SSACC', 'ESACC', 'SBLINK', 'EBLINK']

    default_arff_dtype = 'NUMERIC'
    default_time_dtype = 'INTEGER'  # so that it looks nicer in the arff file (no scientific notation)

    column_names = ['time']
    global_attributes = []  # ARFF attributes; in case of binocular, add extra 'x' and 'y' for mean position
    if not binocular_flag:
        column_names += ['x', 'y', 'pupil size']
        # if both flags are True, velocity comes first
        if velocity_flag:
            column_names += ['velocity x', 'velocity y']
        if resolution_flag:
            column_names += ['resolution x', 'resolution y']

        global_attributes += column_names  # for the monocular case, take it as-is
    else:
        column_names += ['x left', 'y left', 'pupil size left', 'x right', 'y right', 'pupil size right']
        # if both flags are True, velocity comes first
        if velocity_flag:
            column_names += ['velocity x left', 'velocity y left', 'velocity x right', 'velocity y right']
        if resolution_flag:
            column_names += ['resolution x', 'resolution y']

        global_attributes += column_names
        # + mean x/y position of the two eyes
        # place right after 'time' for readability
        global_attributes.insert(1, 'x')
        global_attributes.insert(2, 'y')

        # remember respective indices for coordinates for each eye
        x_left_ind = column_names.index('x left')
        y_left_ind = column_names.index('y left')
        x_right_ind = column_names.index('x right')
        y_right_ind = column_names.index('y right')

    # and the resulting x and y indices within arff object columns (i.e. attributes)
    time_ind = global_attributes.index('time')
    x_ind = global_attributes.index('x')
    y_ind = global_attributes.index('y')

    # +confidence attribute (1 for "normal" samples, 0 for both eyes' tracking lost, 0.5 for one eye lost in case of
    # binocular tracking)
    global_attributes.append('confidence')

    global_attributes = [(attr_name, default_arff_dtype if attr_name != 'time' else default_time_dtype)
                         for attr_name in global_attributes]
    if extract_events_for_eye is not None:
        # will parse events, need one extra attribute
        global_attributes.append((recording_processor.EM_TYPE_ATTRIBUTE_NAME,
                                  recording_processor.EM_TYPE_ARFF_DATA_TYPE))

    if out_folder is None:
        out_folder = os.path.dirname(fname)
    out_folder_full = '{}/{}'.format(out_folder, corresponding_video_name)
    if not os.path.exists(out_folder_full):
        os.makedirs(out_folder_full)
    out_fname = '{}/{}.arff'.format(out_folder_full, os.path.splitext(os.path.basename(fname))[0])

    global_description_lines = []
    global_metadata = OrderedDict([
        ('width_mm', video_width_mm),
        ('height_mm', video_height_mm),
        ('distance_mm', distance_observer_to_screen_mm),
        ('width_px', video_width_px),
        ('height_px', video_height_px)
    ])

    arff_obj = {
        'relation': 'gaze_recording',
        'description': '',
        'data': [],
        'metadata': global_metadata,
        'attributes': global_attributes
    }

    blink_inside_saccade = False  # is current blink inside a saccade
    current_eye_movement = recording_processor.EM_TYPE_DEFAULT_VALUE  # unknown for now

    for line in open(fname):
        line = line.rstrip('\r\n')
        if not line:  # empty line
            continue
        # extract descriptions lines, i.e. '** <some text>'
        if line.startswith('**'):
            line = line.lstrip('* ')  # remove comment signs and spaces
            global_description_lines.append(line)
        else:
            parts = line.split('\t')  # Tab-separated
            try:
                timestamp = int(parts[0])
            except ValueError:
                # this is not a sample line
                # is it an event line?
                if line.split()[0] in convert.EVENT_STRINGS:
                    if extract_events_for_eye is None:
                        continue  # ignore this line
                    # update current eye movement, if eye matches
                    parts = line.split()  # these lines are space-separated
                    event, eye = parts[:2]
                    if eye != extract_events_for_eye:
                        continue  # wrong eye, skip this line
                    # eye matches
                    if event.startswith('S'):  # start of some event, check if no other event is in progress
                        # one exception case, BLINK can start inside SACC
                        if current_eye_movement == 'SACCADE' and event == 'SBLINK':
                            blink_inside_saccade = True
                            current_eye_movement = 'BLINK'
                            continue
                        assert current_eye_movement == recording_processor.EM_TYPE_DEFAULT_VALUE, \
                            'Unexpected {} event in line {}, while previous {} is still active'.\
                            format(event, line, current_eye_movement)
                        if event == 'SFIX':
                            current_eye_movement = 'FIX'
                        elif event == 'SSACC':
                            current_eye_movement = 'SACCADE'
                        elif event == 'SBLINK':
                            current_eye_movement = 'BLINK'
                    else:  # it's an event's end, check that is matches the currently active
                        assert (event == 'EFIX' and current_eye_movement == 'FIX') or \
                               (event == 'ESACC' and current_eye_movement == 'SACCADE') or \
                               (event == 'EBLINK' and current_eye_movement == 'BLINK'), \
                            'Unexpected {} event in line {}, while previous {} is active'.\
                            format(event, line, current_eye_movement)
                        current_eye_movement = recording_processor.EM_TYPE_DEFAULT_VALUE  # reset to default
                        if event == 'EBLINK' and blink_inside_saccade:
                            current_eye_movement = 'SACCADE'
                else:
                    # not a sample, not an event, add it to description to be sure
                    global_description_lines.append(line)
            else:
                # this is a sample
                values = map(soft_float_cast, parts)
                if corneal_reflection_mode:  # last column is the CR status, ignore it
                    values = values[:-1]

                if binocular_flag:
                    # extract mean x and y and insert
                    x_mean = np.nanmean([values[x_left_ind], values[x_right_ind]])
                    y_mean = np.nanmean([values[y_left_ind], values[y_right_ind]])

                    # NB check before inserting!
                    tracking_missing_flags = np.isnan([values[i] for i in [x_left_ind, y_left_ind,
                                                                           x_right_ind, y_right_ind]])
                    values.insert(x_ind, x_mean)
                    values.insert(y_ind, y_mean)

                    if not tracking_missing_flags.any():
                        confidence = 1.0
                    elif not tracking_missing_flags.all():
                        confidence = 0.5
                    else:
                        confidence = 0.0
                else:
                    tracking_missing_flags = np.isnan([values[i] for i in [x_ind, y_ind]])
                    if not tracking_missing_flags.any():
                        confidence = 1.0
                    else:
                        confidence = 0.0

                # append confidence value
                values.append(confidence)

                if extract_events_for_eye is not None:  # append eye movement type
                    values.append(current_eye_movement)

                # check if within restricted time interval, if any (BEFORE applying modifiers)
                if (restrict_time_low is None or values[time_ind] >= restrict_time_low) and \
                        (restrict_time_high is None or values[time_ind] <= restrict_time_high):
                    # apply modifiers
                    values[time_ind] = gaze_time_modifier_microsec(values[time_ind])
                    values[x_ind] = gaze_x_modifier(values[x_ind])
                    values[y_ind] = gaze_y_modifier(values[y_ind])
                    # append to the data list
                    arff_obj['data'].append(values)

    arff_obj['description'] = '\n'.join(global_description_lines)

    ArffHelper.convert_data_to_structured_array(arff_obj)

    with open(out_fname, 'w') as f:
        ArffHelper.dump(arff_obj, f)

    return out_fname


def soft_float_cast(s):
    """
    A method to cast string to floats if possible, otherwise (ex. '.') convert it to NaN
    :param s: string to be cast
    :return: floating point number
    """
    try:
        return float(s)
    except ValueError:
        return np.nan
