import sys
import warnings
import json
import math
import numpy as np

from arff_helper import ArffHelper


class ParameterBunch(object):
    """
    A Namespace alternative, initialized from a dictionary. A similar object is returned by ArgumentParser.
    """
    def __init__(self, adict):
        self.__dict__.update(adict)


def pretty_string(obj):
    """
    Pretty formatting of the object json serialization
    :param obj: object to be serialized
    :return: pretty-printed json serialization of @obj
    """
    return json.dumps(obj, indent=4, separators=(',', ': '))


def pretty_json_dump(obj, fname):
    """
    Pretty formatting of the object json serialization
    :param obj: object to be serialized
    :param fname: output file name
    :return: pretty-printed json serialization of @obj
    """
    with open(fname, 'w') as fp:
        json.dump(obj, fp, indent=4, separators=(',', ': '))


def add_eye_movement_attribute(arff_object):
    """
    Add the EYE_MOVEMENT_TYPE attribute to the @arff_object. If already present, do nothing.
    :param arff_object: arff object
    :return: arff object with added column for eye movement type
    """
    from recording_processor import EM_TYPE_ATTRIBUTE_NAME, EM_TYPE_ARFF_DATA_TYPE, EM_TYPE_DEFAULT_VALUE
    if 'EYE_MOVEMENT_TYPE' not in arff_object['data'].dtype.names:
        ArffHelper.add_column(arff_object,
                              EM_TYPE_ATTRIBUTE_NAME,
                              EM_TYPE_ARFF_DATA_TYPE,
                              EM_TYPE_DEFAULT_VALUE)
    return arff_object


def calculate_ppd(arff_object, skip_consistency_check=False):
    """
    Pixel-per-degree value is computed as an average of pixel-per-degree values for each dimension (X and Y).

    :param arff_object: arff object, i.e. a dictionary that includes the 'metadata' key.
                @METADATA in arff object must include "width_px", "height_px", "distance_mm", "width_mm" and
                "height_mm" keys for successful ppd computation.
    :param skip_consistency_check: if True, will not check that the PPD value for the X axis resembles that of
                                   the Y axis
    :return: pixel per degree.

    """
    # Previous version of @METADATA keys, now obsolete
    calculate_ppd.OBSOLETE_METADATA_KEYS_MAPPING = {
        'PIXELX': ('width_px', lambda val: val),
        'PIXELY': ('height_px', lambda val: val),
        'DIMENSIONX': ('width_mm', lambda val: val * 1e3),
        'DIMENSIONY': ('height_mm', lambda val: val * 1e3),
        'DISTANCE': ('distance_mm', lambda val: val * 1e3)
    }

    for obsolete_key, (new_key, value_modifier) in calculate_ppd.OBSOLETE_METADATA_KEYS_MAPPING.iteritems():
        if obsolete_key in arff_object['metadata'] and new_key not in arff_object['metadata']:
            warnings.warn('Keys {} are obsolete and will not necessarily be supported in future. '
                          'Consider using their more explicit alternatives: {}'
                          .format(calculate_ppd.OBSOLETE_METADATA_KEYS_MAPPING.keys(),
                                  [val[0] for val in calculate_ppd.OBSOLETE_METADATA_KEYS_MAPPING.values()]))
            # replace the key
            arff_object['metadata'][new_key] = value_modifier(arff_object['metadata'].pop(obsolete_key))

    theta_w = 2 * math.atan(arff_object['metadata']['width_mm'] /
                            (2 * arff_object['metadata']['distance_mm'])) * 180. / math.pi
    theta_h = 2 * math.atan(arff_object['metadata']['height_mm'] /
                            (2 * arff_object['metadata']['distance_mm'])) * 180. / math.pi

    ppdx = arff_object['metadata']['width_px'] / theta_w
    ppdy = arff_object['metadata']['height_px'] / theta_h

    ppd_relative_diff_thd = 0.2
    if not skip_consistency_check and abs(ppdx - ppdy) > ppd_relative_diff_thd * (ppdx + ppdy) / 2:
        warnings.warn('Pixel-per-degree values for x-axis and y-axis differ '
                      'by more than {}% in source file {}! '
                      'PPD-x = {}, PPD-y = {}.'.format(ppd_relative_diff_thd * 100,
                                                       arff_object['metadata']['filename'],
                                                       ppdx, ppdy))
    return (ppdx + ppdy) / 2


def get_xy_moving_average(data, window_size, inplace=False):
    """
    Get moving average of 'x', 'y' columns of input data (the moving window is centered around the data point).

    Some data at the beginning and in the end will be left unchanged (where the window does not fit fully).
    Thus the length of offset is equal to (window_size - 1)/2.
    The rest of data will be replaced with central moving average method.

    :param data: structured numpy array that contains columns 'x' and 'y'.
    :param window_size: width of moving average calculation.
    :param inplace: whether to replace input data with processed data (False by default)
    :return: data set with moving average applied to 'x' and 'y' columns.

    """
    assert window_size % 2 == 1, "The @normalization_sliding_window_size_samples parameter is set to {}, but it " \
                                 "has to be odd, so that we can centre the moving window around the current sample.".\
        format(window_size)
    if not inplace:
        data = data.copy()
    offset = (window_size - 1) / 2
    for column in ['x', 'y']:
        res = np.cumsum(data[column], dtype=float)
        res[window_size:] = res[window_size:] - res[:-window_size]
        res = res[window_size - 1:] / window_size
        if offset > 0:
            data[column][offset:-offset] = res
        else:
            data[column][:] = res
    return data


def update_progress(progress, out_stream=sys.stderr, width_count=100):
    if type(progress) == str:
        out_stream.write('\r{}'.format(progress))
        return

    if type(progress) == tuple:
        progress = progress[0] / float(progress[1])
    out_stream.write('\r[{0}] {1:2f}%'.format(('#'*(int(progress * width_count))).ljust(width_count), progress * 100))
