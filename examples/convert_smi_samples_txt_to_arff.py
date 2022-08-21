from collections import OrderedDict
import os

from sp_tool.arff_helper import ArffHelper

# Example usage (from sp_tool source directory)
# $> python
# >> from examples import convert_SMI_Samples_txt_to_arff
# >> convert_SMI_Samples_txt_to_arff.convert(fname='test_data/smi_data_example_recording1_Samples.txt',
#                                            corresponding_video_name='smi_example_video',
#                                            video_width_mm=440, video_height_mm=340,
#                                            distance_observer_to_screen_mm=800,
#                                            video_width_px=1280, video_height_px=720,
#                                            restrict_time_low=2685769677, restrict_time_high=2707351244)


def convert(fname, corresponding_video_name,
            video_width_mm, video_height_mm, distance_observer_to_screen_mm, video_width_px, video_height_px,
            out_folder=None,
            restrict_time_low=None, restrict_time_high=None,
            time_column_name='Time', gaze_x_column_name='L CR1 X [px]', gaze_y_column_name='L CR1 Y [px]',
            gaze_time_modifier_microsec=lambda t: t, gaze_x_modifier=lambda x: x, gaze_y_modifier=lambda y: y
            ):
    """
    Convert (part of) SMI text file (SMI converter from binary format usually names it with a Samples.txt postfix)
    into an ARFF file in respective folder (a subfolder in @out_folder is created for the @corresponding_video_name).
    The @fname basename is preserved, so don't hesitate to execute multiple convert() calls with the same @out_folder.

    The attributes of @time_column_name, @gaze_x_column_name and @gaze_y_column_name are placed on the first 3 spots
    in the resulting arff for readability.


    :param fname: *Samples.txt file to be converted
    :param corresponding_video_name: video name (will create a subdirectory in @out_folder with this name)

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

    :param time_column_name: the name of the column that is used as 'time'
    :param gaze_x_column_name: the name of the column that is used as 'x' coordinate
    :param gaze_y_column_name: the name of the column that is used as 'y' coordinate

    Main fields' modifiers. You can do this as post-processing as well, though (might be more intuitive).
    :param gaze_time_modifier_microsec: a function (one value in, one value out) to modify the timestamp of a gaze
                                        sample to produce microseconds; no change by default
    :param gaze_x_modifier: a function (one value in, one value out) to modify the X coordinate of gaze
                            (ex. if the video was scaled/shifted relative to tracked coordinates and you
                            want to immediately recover the in-video coordinates); no change by default
    :param gaze_y_modifier: a function (one value in, one value out) to modify the Y coordinate of gaze
                            (ex. if the video was scaled/shifted relative to tracked coordinates and you
                            want to immediately recover the in-video coordinates)
    :return: the output file name
    """

    default_arff_dtype = 'NUMERIC'
    default_time_dtype = 'INTEGER'  # so that it looks nicer in the arff file (no scientific notation)
    type_column = None

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
        'attributes': []
    }

    lines = open(fname).readlines()
    data_start_index = None
    for ind, line in enumerate(lines):
        line = line.rstrip('\r\n')
        if not line:
            continue
        # might be useful to extract lines line '## Some Key Word: <tab-separated value list>'
        # here we just add all comment lines as the file description in ARFF format too
        if line.startswith('##'):
            line = line.lstrip('# ')  # remove comment signs and spaces
            global_description_lines.append(line)
        else:
            # extract the required attributes, store the rest with default type too
            format_line = line.split('\t')
            for attribute in format_line:
                if attribute == 'Type':
                    type_column = format_line.index('Type')
                    # no need to add this attribute
                elif attribute == time_column_name:
                    arff_obj['attributes'].append(('time', default_time_dtype))
                elif attribute == gaze_x_column_name:
                    arff_obj['attributes'].append(('x', default_arff_dtype))
                elif attribute == gaze_y_column_name:
                    arff_obj['attributes'].append(('y', default_arff_dtype))
                else:
                    arff_obj['attributes'].append((attribute, default_arff_dtype))
            # actual data can start next line
            data_start_index = ind + 1
            break
    arff_obj['description'] = '\n'.join(global_description_lines)

    time_index = arff_obj['attributes'].index(('time', 'INTEGER'))
    x_index = arff_obj['attributes'].index(('x', 'NUMERIC'))
    y_index = arff_obj['attributes'].index(('y', 'NUMERIC'))

    original_attributes = arff_obj['attributes'][:]  # copy the old list
    # put t-x-y as first attributes in the output
    reordered_attributes = [
        arff_obj['attributes'][time_index],
        arff_obj['attributes'][x_index],
        arff_obj['attributes'][y_index]
    ]
    # add all other attributes
    reordered_attributes += [a for ind, a in enumerate(arff_obj['attributes']) if ind not in {time_index,
                                                                                              x_index,
                                                                                              y_index}]
    arff_obj['attributes'] = reordered_attributes

    for data_line in lines[data_start_index:]:
        values = data_line.strip().split('\t')

        line_type = values[type_column]
        if line_type == 'SMP':  # sample
            values = list(map(float, values[:type_column] + values[type_column + 1:]))  # all values except type
            assert len(values) == len(original_attributes), \
                'Attributes list does not match loaded values: "{}" vs "{}"'.format(arff_obj['attributes'],
                                                                                    values)
            if (restrict_time_low is None or values[time_index] >= restrict_time_low) and \
                    (restrict_time_high is None or values[time_index] <= restrict_time_high):

                # modify, if necessary
                values[time_index] = gaze_time_modifier_microsec(values[time_index])
                values[x_index] = gaze_x_modifier(values[x_index])
                values[y_index] = gaze_y_modifier(values[y_index])

                # reorder values to match new attributes
                reordered_values = [values[arff_obj['attributes'].index(a)] for a in original_attributes]
                arff_obj['data'].append(reordered_values)

    ArffHelper.convert_data_to_structured_array(arff_obj)

    with open(out_fname, 'w') as f:
        ArffHelper.dump(arff_obj, f)

    return out_fname



