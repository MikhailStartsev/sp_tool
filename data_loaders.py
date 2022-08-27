import os

import numpy as np
from functools import wraps
from collections import OrderedDict
import inspect

from sp_tool.arff_helper import ArffHelper
import util

EM_VALUE_MAPPING_DEFAULT = {
    0: 'UNKNOWN',
    1: 'FIX',
    2: 'SACCADE',
    3: 'SP',
    4: 'NOISE',
    10: 'PSO',
    11: 'BLINK'
}


def write_arff_result(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        output_arff_fname = inspect.getcallargs(func, *args, **kwargs).get('output_arff_fname', None)
        arff_obj = func(*args, **kwargs)

        if output_arff_fname is not None:
            with open(output_arff_fname, 'w') as arff_out:
                ArffHelper.dump(arff_obj, arff_out)
        return arff_obj
    return wrapper


def load_ARFF_as_arff_object(fname, eye_movement_type_attribute=None, eye_movement_type_mapping_dict=None):
    """
    Load data from ARFF file format (with %@METADATA comments possible, see [1]).
    We expect and verify that the arff file in question has the columns 'time', 'x' and 'y'
    (for the timestamp, x and y position of the gaze point, respectively).

    [1] http://ieeexplore.ieee.org/abstract/document/7851169/
    :param fname: name of the .arff file
    :param eye_movement_type_attribute: the attribute that should be treated as an indication
                                        of eye movement type, optional;
                                        should be either a string (name of the attribute), or True, in which case
                                        it is substituted by the 'EYE_MOVEMENT_TYPE' string
    :param eye_movement_type_mapping_dict: a dictionary that is used to convert values in column
                                           @eye_movement_type_attribute to values in the following set:
                                           ['UNKNOWN', 'FIX', 'SACCADE', 'SP', 'NOISE', 'BLINK', 'NOISE_CLUSTER']
                                           (as defined by recording_processor.py)
    :return: an arff object
    """
    # EM_VALUE_MAPPING_DEFAULT is the inverse of the dictionary used in evaluation.py.
    # It can be used (with @eye_movement_type_mapping_dict='default'), for instance, to load the files where
    # different eye movements are labelled by numerical values rather than by categorical (i.e. strings), due to
    # arff file implementation in the framework that produced the labels, or some other reason.
    # These values correspond to the ones used in our hand-labelling tool [1].

    arff_obj = ArffHelper.load(open(fname))
    # validate that we have all the essential data
    assert all([attr in arff_obj['data'].dtype.names for attr in ['time', 'x', 'y']]), \
        'File {} must contain at least "time", "x" and "y" columns'.format(fname)

    if eye_movement_type_attribute is not None:
        from recording_processor import EM_TYPE_ARFF_DATA_TYPE, EM_TYPE_ATTRIBUTE_NAME
        if eye_movement_type_attribute is True:
            eye_movement_type_attribute = EM_TYPE_ATTRIBUTE_NAME

        assert eye_movement_type_attribute in arff_obj['data'].dtype.names, \
            'Attribute {} is not present in the arff structure from file {}'.format(eye_movement_type_attribute,
                                                                                    fname)
        # add the dedicated eye movement type column
        arff_obj = util.add_eye_movement_attribute(arff_obj)
        if eye_movement_type_mapping_dict is None:
            # Check if the column is not yet of the right format.
            # Only need to do this if the attribute is numerical, not categorical!
            if arff_obj['data'][eye_movement_type_attribute].dtype.type is not np.string_:
                correct_flag = all([item in EM_TYPE_ARFF_DATA_TYPE
                                    for item in arff_obj['data'][eye_movement_type_attribute]])
            else:  # nothing to do here, already a categorical attribute
                correct_flag = True

            if correct_flag:
                # already the perfect values in the respective column, just put the same values in the special column
                arff_obj['data']['EYE_MOVEMENT_TYPE'] = arff_obj['data'][eye_movement_type_attribute]
                return arff_obj
            else:
                # if None, act as 'default', if needed
                eye_movement_type_mapping_dict = EM_VALUE_MAPPING_DEFAULT
        elif eye_movement_type_mapping_dict == 'default':
            eye_movement_type_mapping_dict = EM_VALUE_MAPPING_DEFAULT

        assert isinstance(eye_movement_type_mapping_dict, dict), 'Argument @eye_movement_type_mapping_dict must be ' \
                                                                 'either a dict, or None, or a string "default"'
        assert all([v in EM_TYPE_ARFF_DATA_TYPE for v in list(eye_movement_type_mapping_dict.values())]), \
            'All the values of the provided dictionary must be one of the following: {}'.format(EM_TYPE_ARFF_DATA_TYPE)
        # now map using the dictionary
        original_values = arff_obj['data'][eye_movement_type_attribute]
        mapped_values = [eye_movement_type_mapping_dict[item] for item in original_values]
        arff_obj['data']['EYE_MOVEMENT_TYPE'] = mapped_values

    arff_obj['metadata']['filename'] = fname
    return arff_obj


@write_arff_result
def load_DSF_coord_as_arff_object(fname, output_arff_fname=None):
    """
    Load data from the given input .coord file and return an arff object.
    This is a "model" function for writing new data adapters. To create a similarly-functioning method,
    one would need to parse the file under @fname to extract an an arff object (dictionary with special keys)
    ofr the following structure:

    arff_obj = {
        'relation': 'gaze_recording',
        'description': '',
        'data': [],
        'metadata': {},
        'attributes': [('time', 'INTEGER'),
                       ('x', 'NUMERIC'),
                       ('y', 'NUMERIC'),
                       ('confidence', 'NUMERIC')]},
    and fill in its fields.

    'data' should first contain a numpy list of lists (the latter lists should be of the same length as 'attributes'.
    'description' is just a string that gets put into the beginning of the file.
    'metadata' is a dictionary, where the following keys are needed later on:
        - "width_px", "height_px" - pixel dimensions of the video
        - "width_mm", "height_mm" - physical dimensions of the video (in millimeters)
        - "distance_mm" - distance between the observer's eyes and the monitor (in millimeters)
    'attributes' (if additional ones are required) is a list of tuples, each tuple consisting of 2 elements:
        - attribute name
        - attribute type, can be INTEGER (=int64), NUMERIC (=float32), REAL (=double), or a list of strings, which
          means it is a categorical attribute and only these values are accepted.

    After 'data' is filled with appropriate lists of values, call
    >> arff_obj = ArffHelper.convert_data_to_structured_array(arff_obj)
    to (unsurprisingly) convert the data in @arff_obj['data'] into a structured numpy array for easier data access.


    :param fname: name of .coord file.
    :param output_arff_fname: if desired, this function can also convert the input .coord file into an .arff file,
                              that can be further used within this framework as well.
    :return: an arff object with keywords:
             "@RELATION, @DESCRIPTION, @DATA, @METADATA, @ATTRIBUTES".
    """

    load_DSF_coord_as_arff_object.COMMENT_PREFIX = '#'
    # the 'gaze ... ...' line has this many "fields" (defines the video resolution)
    load_DSF_coord_as_arff_object.GAZE_FORMAT_FIELD_COUNT = 3
    # Samples are in lines that look like <timestamp> <x> <y> <confidence>.
    # In case of binocular tracking, these are the mean coordinates of the two eyes anyway.
    load_DSF_coord_as_arff_object.GAZE_SAMPLE_FIELDS = 4

    if not os.path.isfile(fname):
        raise ValueError("No such .coord file named '{}' or incorrect input format of file name".format(fname))

    arff_obj = {
        'relation': 'gaze_recording',
        'description': [],
        'data': [],
        'metadata': OrderedDict(),
        'attributes': [('time', 'INTEGER'),
                       ('x', 'NUMERIC'),
                       ('y', 'NUMERIC'),
                       ('confidence', 'NUMERIC')]}
    description = []

    for line in open(fname):
        line = line.rstrip('\n ')
        if line.startswith(load_DSF_coord_as_arff_object.COMMENT_PREFIX):
            description.append(line[len(load_DSF_coord_as_arff_object.COMMENT_PREFIX):])
            continue

        try:
            ll = line.split()
            # cut out the first needed values (t, x, y, confidence), even if binocular tracking .coord file
            ll = list(map(float, ll))[:load_DSF_coord_as_arff_object.GAZE_SAMPLE_FIELDS]
            arff_obj['data'].append(ll)
        except ValueError:
            if line.startswith('gaze'):
                words = line.split()
                # This line should the following format:
                # gaze <video width in pixels> <video height in px>
                if len(words) == load_DSF_coord_as_arff_object.GAZE_FORMAT_FIELD_COUNT:
                    arff_obj['metadata']['width_px'] = float(words[1])
                    arff_obj['metadata']['height_px'] = float(words[2])
                else:
                    raise ValueError("Incorrect gaze data format in file {}. "
                                     "Correct format should be 'gaze <width_in_pixels> <height_in_pixels>'".
                                     format(fname))
            elif line.startswith('geometry'):
                words = line.split()
                # This line should the following format:
                # geometry <property_name_1> <value in meters> <property_name_2> <value in meters> ...
                # So we deem every second field as property name or value in meters, respectively.
                # We convert the values to mm.
                for i in range(1, len(words), 2):
                    key_mm = '{}_mm'.format(words[i])
                    value_mm = float(words[i + 1]) * 1e3
                    arff_obj['metadata'][key_mm] = value_mm
            continue
    arff_obj['metadata']['filename'] = fname

    arff_obj['description'] = '\n'.join(description)
    arff_obj = ArffHelper.convert_data_to_structured_array(arff_obj)

    return arff_obj
