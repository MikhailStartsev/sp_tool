#!/usr/bin/env python
import tempfile
import warnings
import os
import sys
import argparse
import glob
from typing import Iterable
from collections import defaultdict, OrderedDict
import json
import traceback

from saccade_detector import SaccadeDetector
from blink_detector import BlinkDetector
from fixation_detector import FixationDetector
from recording_processor import RecordingProcessor
from sp_detector import SmoothPursuitDetector
from arff_helper import ArffHelper
import util

# This file comprises a very flexible console interface (see `python run_detection.py -h`)
# as well as a programmatic interface (through a pair of function calls: create_parameters() and run_detection() ).
# For example,
#   >> import run_detection
#   >> params = run_detection.create_parameters(input_folder='DESIRED/PATH/TO/DATA', verbose=True)
#   >> run_detection.run_detection(params)
# OR
# $> python run_detection.py --input-folder DESIRED/PATH/TO/DATA --verbose
#
# The run_detection method takes a specific two-layered dictionary of parameters as input, so you could load them
# from a configuration file (like the example default_parameters.conf.json 
# file provided, though you don't need to set values to parameters you want to leave default, they will be generated 
# automatically; the example default parameters file mentions all the possible parameter names you can set), 
# or create using a create_parameters() method.
#
# This should allow you to configure all the detectors that are used in this pipeline, as well as set desired
# input/output locations.
#
# Notes on the detectors:
# (1) The saccade and blink detectors are quite robust together, their parameters are probably best left default, unless
# there is a specific reason to change them.
# (2) The balance between precision and recall of SP is currently shifted towards precision: we want to be sure that
# what we almost as SP, mostly is, while detecting a decent part of the real SP. If your priorities diverge from this,
# you should adjust the parameters of the SmoothPursuitDetector, relaxing the thresholds a bit, as well as
# the parameters of the FixationDetector, since most of the recall loss actually happens there.
# There you can try to relax the "prefiltering_interval_spread_threshold_degrees", "speed_threshold_degrees_per_sec" and
# "min_sp_duration_microsec" parameters.

# Here starts the main "programmatic interface" of this module
# For console interface help, try `python run_detection.py --help`.


def run_detection(params):
    """
    Run the entire detection pipeline with given parameters.
    :param params: A two-level dictionary (just like create_parameters_from_args() would return).
                   The only required parameter is @params['GeneralArguments']['input_folder'], which should point
                   to a folder with raw gaze data. The data is assumed to be stored in the following way:
                     (1) for each movie (clip) there should be a separate subdirectory in the input_folder
                     (2) inside these subdirectories all the files with the extension of
                     @params['GeneralArguments']['gaze_extension'] (.coord by default) represent a recording for one
                     observer each.
                   If your data does not get loaded, maybe the appropriate data loader does not get called. You can
                   fix this (provided that the suitable data loader exists in data_loaders.py) by setting
                   @params['GeneralArguments']['input_data_type'] to the correct value (for correspondence see
                   the keys of RecordingProcessor._format_loaders).

                   To summarize, a minimalistic input to run detection with default parameters on your dataset
                   (let's assume you have converted the data to .arff format) would be:

                   run_detection({'GeneralArguments': {'input_folder': 'PATH/TO/YOUR/DATA/FOLDER',
                                                       'gaze_extension': '.arff'}})
    :return: path to results folder
    """
    # make a defaultdict  out of @parameters so that we could always access its first-level keys
    params_default_first_level = defaultdict(dict)
    params_default_first_level.update(params)
    params = params_default_first_level

    verbose = params['GeneralArguments'].get('verbose', False)

    out_folder = params['GeneralArguments'].get('output_folder')
    if out_folder is None:
        out_folder = tempfile.mkdtemp(prefix='sp_tool_')
        warnings.warn('No output folder provided, using {}'.format(out_folder))
    if verbose:
        print('Outputs will be written to folder', out_folder, file=sys.stderr)

    saccade_detector = SaccadeDetector(**params['SaccadeDetector'])
    blink_detector = BlinkDetector(**params['BlinkDetector'])
    fixation_detector = FixationDetector(**params['FixationDetector'])

    recording_processor = RecordingProcessor(saccade_detector=saccade_detector,
                                             blink_detector=blink_detector,
                                             fixation_detector=fixation_detector)

    sp_detector = SmoothPursuitDetector(**params['SmoothPursuitDetector'])

    # The next lines deal with identifying the names of the video clips used for the eye tracking experiment.
    # Can be initialized in various ways, here we just get all video paths be regex and cut off everything that
    # is not needed.
    #
    #
    in_folder = params['GeneralArguments'].get('input_folder')
    if not in_folder:
        raise ValueError('\'input_folder\' is a required parameter of the \'GeneralArguments\' group in @params!')
    folder_names = sorted(glob.glob('{}/*/'.format(in_folder)))  # getting all the folders of the input folder
    # extract names from path
    if not folder_names and verbose:
        print('No subfolders found under "{}"'.format(in_folder), file=sys.stderr)
    folder_names = [os.path.basename(folder.rstrip(os.path.sep)) for folder in folder_names]

    movies = params['GeneralArguments'].get('movies')
    if movies:  # not empty, restrict to these folders only
        movies = set(movies)
        folder_names = [fn for fn in folder_names if fn in movies]

    if verbose:
        print('Working with movies:', folder_names, file=sys.stderr)

    # data files extension
    gaze_pattern = params['GeneralArguments'].get('gaze_file_pattern', '*.coord')
    if '*' not in gaze_pattern:
        gaze_pattern = '*' + gaze_pattern

    for movie in folder_names:
        full_out_folder = '{}/{}/'.format(out_folder, movie)
        if not os.path.exists(full_out_folder):
            os.makedirs(full_out_folder)
        if verbose:
            print('Started processing for {},'.format(movie), 'results will appear in', full_out_folder, file=sys.stderr)

        # The next lines load the data files of the recording with one particular movie.
        # To do this, here we provide a regex that includes all the .{extension} files in the respective folder.
        #
        #
        gaze_data_files = sorted(glob.glob('{}/{}/{}'.format(in_folder, movie, gaze_pattern)))
        if len(gaze_data_files) == 0:
            print('Found 0 files with this pattern: "{}". Omitting this directory.'.format(
                '{}/{}/{}'.format(in_folder, movie, gaze_pattern)
            ), file=sys.stderr)
            continue
        try:
            # The next line loads the data, labels saccades, blinks and fixations.
            gaze_points_list = recording_processor.load_multiple_recordings(
                gaze_data_files, verbose=verbose, data_format=params['GeneralArguments'].get('input_data_type'))
            # This will label the smooth pursuits
            if verbose:
                print('Saccades/blinks/fixations are detected, starting SP detection.', file=sys.stderr)
            classified_gaze_points = sp_detector.detect(gaze_points_list)

            # Now just dump the resulting structure into .arff files in the respective subdirectory of the @out_folder
            for file_name, arff_data in zip(gaze_data_files, classified_gaze_points):
                output_file_name = os.path.splitext(os.path.basename(file_name))[0]
                ArffHelper.dump(arff_data, open(
                    '{}/{}.arff'.format(full_out_folder, output_file_name), 'w')).close()
        except Exception as e:
            print('Had to skip {} due to an error "{}"'.format(movie, e), file=sys.stderr)
            print(''.join(traceback.format_exception(None, e, e.__traceback__)))
    return out_folder


def create_parameters(config_file=None,
                      input_folder=None, gaze_file_pattern=None, output_folder=None,
                      ignore_unused_arguments=False,
                      **kwargs):
    """
    A way pf creating parameters without console arguments.
    Intended use (for example):
    >> params = create_parameters(input_folder='/PATH/TO/YOUR/DATA', gaze_file_pattern='*.YOUR_EXT', ...)
    or
    >> params = create_parameters(config_file='default_parameters.conf.json',
                                  input_folder='/PATH/TO/YOUR/DATA',
                                  verbose=True)

    >> run_detection(params)

    :param config_file: a path to a configuration file (like a default configuration file);
                        either @config_file (with an "input_folder" parameter) or @input_folder (or both)
                        *must* be provided (otherwise we don't know where the data is located)
    :param input_folder: folder with gaze data (--input-folder in console arguments).
                         This folder is assumed to have subfolders that correspond to videos for which recordings
                         were made. Each such subdirectory should contain gaze files (one file per observer).
                         If provided, overrides the "input_folder" value of the "GeneralParameters" in @config_file.
                         If not, @config_file with a valid 'input_file" must be provided!
    :param gaze_file_pattern: a wildcard pattern of gaze files.
                              If provided, overrides the "gaze_file_pattern" value of the "GeneralParameters" in
                              @config_file. If not, @config_file with a valid 'gaze_file_pattern' must be provided!
    :param output_folder: where to write processed files (respective subdirectories for individual videos are created
                          automatically)
    :param ignore_unused_arguments: passed into create_parameters_from_args(), should mostly be kept as False!
    :param kwargs: additional arguments; all arguments in the console interface are supported (with the key of the
                   longest argument name, '_' characters instead of '-' characters, and no '--' in the beginning of
                   the name)
    :return: parameter dictionary, a two-level OrderedDict structure
    """
    # create a namespace as if after argparse and return create_parameters_from_args()
    args_dict = kwargs

    # do not include None values
    dict_candidates = list(zip(['config_file', 'input_folder', 'gaze_file_pattern', 'output_folder'],
                          [config_file, input_folder, gaze_file_pattern, output_folder]))
    args_dict.update({key: value for key, value in dict_candidates if value is not None})

    bunch = util.ParameterBunch(args_dict)
    return create_parameters_from_args(bunch, ignore_unused_arguments=ignore_unused_arguments)

# The following methods are less insightful, and present more or less the internal structure of parameters we create
# to combine a diverse console interface with a usable and more intuitive parameters' structure format.


def create_parameters_description():
    """
    Creates a structure for creating an argument parser and processing its results.
    The structure is represented with a dictionary with first-level keys for group names
    (if group name is empty, this is just an argument, not attributed to any group; this function however, places
    such arguments under GeneralArguments group).

    For each group name, the value of this key in the resulting dictionary is a list of possible arguments.
    Each argument is in turn represented by a dictionary. This dictionary has the following keys:
        - 'main_argument_name', i.e. main name of the argument; this is a "long name" using which the variable name
                                is created in the parser's namespace
        - 'argument_name_options', i.e. all other options of writing the argument name, can include shorter names
        - 'parameter_name', i.e. the name of the parameter *of the respective class initializer or function*. When
                            converting from the parsed arguments to function calls' parameters, this name will be used
        - 'soft_type', i.e. to which type to *try* to cast, compliance is not enforced (an optional key);
                       this is an addition to the `type` argument of ArgumentParser.add_argument call (which can be
                       specified under 'kwargs' key). Can be a list of multiple values, then will try to cast to
                       all of them in turn.
        - 'help', i.e. the help string for this argument
        - 'kwargs', i.e. a dictionary of keyword arguments to be passed into the respective add_argument call
                    when creating the parser
    :return: description dictionary (an OrderedDict)

    """
    description = OrderedDict()
    description['GeneralArguments'] = [
        {
            'main_argument_name': '--config-file',
            'argument_name_options': ['--config'],
            'parameter_name': 'config_file',
            'help': """A json-encoded configuration file, in which one can specify the parameters
                       for all detectors in use as well as some general parameters for the whole run.
                       The encoded object should therefore be a dictionary,
                       with possible top-level keys 'GeneralArguments' (general parameters, not relevant
                       to a detector class), 'SaccadeDetector', 'BlinkDetector', 'FixationDetector'
                       and 'SmoothPursuitDetector'.

                       The value for each of the present keys should in turn be a dictionary with keys
                       identical to the longest argument names below, without the eye movement name prefix.
                       An example (and equivalent to default parameters) configuration file is provided
                       in default_parameters.conf.json and includes all possible keys.

                       In your custom configuration file you do not have to specify any the parameter values,
                       missing keys will be considered to have the default value.

                       For default values, you can consult the respective classes' __init__ methods in
                       saccade_detector.py, blink_detector.py, fixation_detector.py and sp_detector.py.


                       Values given through the console interface override the ones in the config file.""",
            'kwargs': {}
        },
        {
            'main_argument_name': '--input-folder',
            'argument_name_options': ['--in'],
            'parameter_name': 'input_folder',
            'help': 'From where to load the gaze points data. If absent, must be present in --config-file file. '
                    'This folder is assumed to have subfolders that correspond to videos, for which recordings '
                    'were made. Each such subdirectory should contain gaze files (one file per observer).',
            'kwargs': {}
        },
        {
            'main_argument_name': '--gaze-file-pattern',
            'argument_name_options': ['--pattern'],
            'parameter_name': 'gaze_file_pattern',
            'help': 'Will look for such files in all subdirectories of --input-folder. '
                    'For GazeCom, \'*.arff\' is a recommended value (or \'*.coord\', if dealing with original dataset files). '
                    'One can use this parameter to match some name pattern as well (not just the file extension), '
                    'for example with \'*_needed_files_*.arff\'. \n'
                    'If no wildcard symbol is found in the provided string, it is assumed to be just the file name '
                    'suffix, so it will be prepended with a wildcard symbol (i.e. ".coord" will become "*.coord").',
            'kwargs': {}
        },
        {
            'main_argument_name': '--input-data-type',
            'argument_name_options': ['--type'],
            'parameter_name': 'input_data_type',
            'help': 'Type of data loader to use (if not specified, will try to detect automatically)',
            'kwargs': {'choices': ['DSF', 'ARFF', 'labelled ARFF']}
        },
        {
            'main_argument_name': '--verbose',
            'argument_name_options': ['-v'],
            'parameter_name': 'verbose',
            'default': None,
            'help': 'Whether to output some information about the progress of the run to STDERR',
            'kwargs': {'action': 'store_const', 'const': True}  # only like this can support the default of None
                                                                # (not to override the config all the time
                                                                # with a missing value)
        },
        {
            'main_argument_name': '--movies',
            'argument_name_options': ['-m'],
            'parameter_name': 'movies',
            'help': 'Which movies out of the input folder to use (might be useful for train/test split). '
                    'The gaze data is supposed to be put under respective directories in the input folder. '
                    'If none are given, all available ones are used.',
            'kwargs': {'nargs': '+', 'default': None}
        },
        {
            'main_argument_name': '--output-folder',
            'argument_name_options': ['--out'],
            'parameter_name': 'output_folder',
            'help': 'Where to output the resulting labelled data (if empty, will create a new temporary directory)',
            'kwargs': {}
        },
    ]

    description['SaccadeDetector'] = [
        {
            'main_argument_name': '--tolerance',
            'argument_name_options': ['--tol'],
            'parameter_name': 'tolerance',
            'help': 'The relative size of the area outside the screen that is still considered to be legal',
            'kwargs': {'type': float}
        },
        {
            'main_argument_name': '--threshold-onset-fast-degree-per-sec',
            'argument_name_options': ['--threshold-onset-fast'],
            'parameter_name': 'threshold_onset_fast_degree_per_sec',
            'help': 'Threshold for initialization of saccade detection, in degrees per second',
            'kwargs': {'type': float}
        },
        {
            'main_argument_name': '--threshold-onset-slow-degree-per-sec',
            'argument_name_options': ['--threshold-onset-slow'],
            'parameter_name': 'threshold_onset_slow_degree_per_sec',
            'help': 'A slower threshold for saccade onset detection, in degrees per second',
            'kwargs': {'type': float}
        },
        {
            'main_argument_name': '--threshold-offset-degree-per-sec',
            'argument_name_options': ['--threshold-offset'],
            'parameter_name': 'threshold_offset_degree_per_sec',
            'help': 'Threshold for saccade offset detection, in degrees per second',
            'kwargs': {'type': float}
        },
        {
            'main_argument_name': '--max-speed-degree-per-sec',
            'argument_name_options': ['--max-speed'],
            'parameter_name': 'max_speed_degree_per_sec',
            'help': 'Maximum speed of saccadic eye movements',
            'kwargs': {'type': float}
        },
        {
            'main_argument_name': '--min-duration-microsec',
            'argument_name_options': ['--min-duration'],
            'parameter_name': 'min_duration_microsec',
            'help': 'Minimal saccade duration threshold',
            'kwargs': {'type': float}
        },
        {
            'main_argument_name': '--max-duration-microsec',
            'argument_name_options': ['--max-duration'],
            'parameter_name': 'max_duration_microsec',
            'help': 'Maximal saccade duration threshold',
            'kwargs': {'type': float}
        },
        {
            'main_argument_name': '--velocity-integral-interval-microsec',
            'argument_name_options': ['--velocity-integral-interval'],
            'parameter_name': 'velocity_integral_interval_microsec',
            'help': 'Interval duration, over which to integrate velocity computation.',
            'kwargs': {'type': float}
        },
    ]

    description['BlinkDetector'] = [
        {
            'main_argument_name': '--max-distance-to-saccade-microsec',
            'argument_name_options': ['--max-distance-to-saccade'],
            'parameter_name': 'max_distance_to_saccade_microsec',
            'help': 'Threshold for distance from a definite blink to a nearby saccade, which will be marked as blink '
                    'as well.',
            'kwargs': {'type': float}
        },
    ]

    description['FixationDetector'] = [
        {
            'main_argument_name': '--prefiltering-interval-spread-threshold-degrees',
            'argument_name_options': ['--prefiltering-interval-spread-threshold'],
            'parameter_name': 'prefiltering_interval_spread_threshold_degrees',
            'help': 'All the intersaccadic intervals shorter than this will be deemed fixations',
            'kwargs': {'type': float}
        },
        {
            'main_argument_name': '--min-sp-duration-microsec',
            'argument_name_options': ['--min-sp-duration'],
            'parameter_name': 'min_sp_duration_microsec',
            'help': 'Minimal duration of a potential SP candidate (fast-moving samples shorter than this threshold '
                    'are labelled as noise)',
            'kwargs': {'type': float}
        },
        {
            'main_argument_name': '--sliding-window-width-microsec',
            'argument_name_options': ['--sliding-window-width'],
            'parameter_name': 'sliding_window_width_microsec',
            'help': 'Sliding window for coordinates smoothing',
            'kwargs': {'type': float}
        },
        {
            'main_argument_name': '--normalization-sliding-window-size-samples',
            'argument_name_options': ['--normalization-sliding-window'],
            'parameter_name': 'normalization_sliding_window_size_samples',
            'help': 'A moving average sliding window size (to normalize the data)',
            'kwargs': {'type': int}
        },
        {
            'main_argument_name': '--speed-threshold-degrees-per-sec',
            'argument_name_options': ['--speed-threshold'],
            'parameter_name': 'speed_threshold_degrees_per_sec',
            'help': 'Biggest plausible speed for a noisy fixation',
            'kwargs': {'type': float}
        },
        {
            'main_argument_name': '--sliding-window-criterion',
            'argument_name_options': ['--sliding-window'],
            'parameter_name': 'sliding_window_criterion',
            'help': 'Defines the way we check the samples with the sliding_window_criterion threshold: '
                    'either compute the average speed in the current window, or get the spread of '
                    'the gaze points (i.e. biggest XY bounding box side), divided by the duration',
            'kwargs': {'choices': ['speed', 'spread']}
        },
        {
            'main_argument_name': '--intersaccadic-interval-duration-threshold-microsec',
            'argument_name_options': ['--intersaccadic-interval-duration-threshold'],
            'parameter_name': 'intersaccadic_interval_duration_threshold_microsec',
            'help': 'Minimal size of the intersaccadic interval to apply the step with the moving average analysis',
            'kwargs': {'type': float}
        },
    ]

    description['SmoothPursuitDetector'] = [
        # a mutually exclusive group
        [
            {
                'main_argument_name': '--min-pts',
                'argument_name_options': [],
                'parameter_name': 'min_pts',
                'soft_type': int,
                'help': 'An integer indicating the minimum number of points required to form a core point\'s '
                        'neighbourhood, or a string \'num_observers\' (meaning that the actual number of observers '
                        'for each movie will be substituted, depending on the data set provided).\n'
                        'This option is mutually exclusive with --min-observers.',
                'kwargs': {}
            },
            {
                'main_argument_name': '--min-observers',
                'argument_name_options': [],
                'parameter_name': 'min_observers',
                # first try casting to int, then to float (since int cast will fail for a float)
                'soft_type': [int, float],
                'help': 'Either a floating point in [0.0; 1.0] range (indicating the share of all the present '
                        'observers per movie) or int [2; +\inf) (indicating the absolute threshold for '
                        'observer count in the core point\'s neighbourhood).\n'
                        'This option is mutually exclusive with --min-pts.',
                'kwargs': {}
            }
        ],
        {
            'main_argument_name': '--eps-deg',
            'argument_name_options': ['--eps'],
            'parameter_name': 'eps_deg',
            'help': 'Spatial Euclidean distance threshold that defines the neighbourhood in the XY-plane',
            'kwargs': {'type': float}
        },
        {
            'main_argument_name': '--time-slice-microsec',
            'argument_name_options': ['--time-slice'],
            'parameter_name': 'time_slice_microsec',
            'help': 'Width of the time slice that defines the size of the neighbourhood on the time axis.',
            'kwargs': {'type': float}
        },
    ]

    return description


def parse_args():
    parser = argparse.ArgumentParser('SP detection tool main console interface')

    param_description = create_parameters_description()
    for group in param_description:
        if not group:
            # arguments without group
            grp = parser  # fake group
        else:
            grp = parser.add_argument_group(group)

        for item in param_description[group]:
            if isinstance(item, dict):
                grp.add_argument(item['main_argument_name'], *item['argument_name_options'],
                                 help=item['help'],
                                 **item['kwargs'])
            else:
                # this is a mutually exclusive group then
                assert isinstance(item, Iterable)
                excl_grp = grp.add_mutually_exclusive_group()
                for excl_item in item:
                    excl_grp.add_argument(excl_item['main_argument_name'], *excl_item['argument_name_options'],
                                          help=excl_item['help'],
                                          **excl_item['kwargs'])

    return parser.parse_args()


def soft_cast(value, data_type):
    if data_type is None:
        return value
    # make it always a list of types
    if not isinstance(data_type, Iterable):
        data_type = [data_type]

    # if already of one of the soft cast types, just keep it as-is
    for candidate_type in data_type:
        if isinstance(value, candidate_type):
            return value

    for candidate_type in data_type:
        try:
            value = candidate_type(value)
        except ValueError:
            pass
        else:
            # cast successful, break
            break
    return value


def create_parameters_from_args(parsed_args, ignore_unused_arguments=False):
    """
    Parse the result of args into usable group-separated structure. None values (i.e. there was no such argument
    provided) are ignored.
    :param parsed_args:result of argument parsing, a namespace
    :param ignore_unused_arguments: if False, an exception will be thrown, if some of the @parsed_args did not make
                                    it into the returned parameters (i.e. the names did not match any expected ones).

                                    It is advised to keep @ignore_unused_arguments False, since this will prevent
                                    errors due to typos in parameter names (ex. setting @parsed_args.inpuIt_folder
                                    to the desired value, which will be otherwise ignored, while
                                    the @parsed_args.input_folder is loaded from the default config file)!

    :return: a two-level dictionary (OrderedDict) with first-level keys consisting of group names,
    second-level keys -- of parameter names
    """
    param_description = create_parameters_description()
    # get a dictionary view on this
    args_dict = vars(parsed_args).copy()
    # clean up the args, i.e. remove None values so that they do not remain 'untouched' until the end
    args_dict = {k: v for k, v in list(args_dict.items()) if v is not None}

    # we will pop the arguments from args_dict to ensure no not-used parameters are passed

    res_params = OrderedDict()
    # if config file present, load from there
    if args_dict.get('config_file') is not None:
        res_params = json.load(open(args_dict.pop('config_file')),
                               object_pairs_hook=OrderedDict)

    # Hardcoded condition: if the terminal arguments have a value for either --min-pts or --min-observers, remove
    # these values from the preloaded parameters from config file (otherwise conflicts will not be pleasant)
    if args_dict.get('min_pts') is not None or args_dict.get('min_observers') is not None:
        if 'SmoothPursuitDetector' in res_params:
            deleted = False
            for key in ['min_pts', 'min_observers']:
                if key in res_params['SmoothPursuitDetector']:
                    res_params['SmoothPursuitDetector'].pop(key)
                    deleted = True
            if deleted:
                warnings.warn('Found --min-pts or --min-observers in terminal arguments, '
                              'will ignore these values from config!')

    for group in param_description:
        if group not in res_params:
            res_params[group] = OrderedDict()
        for item in param_description[group]:
            if isinstance(item, dict):
                item_vars_name = item['main_argument_name'].lstrip('-').replace('-', '_')
                if args_dict.get(item_vars_name) is not None:
                    res_params[group][item['parameter_name']] = soft_cast(args_dict.pop(item_vars_name),
                                                                          item.get('soft_type', None))
            else:
                # this is a mutually exclusive group, find the one that is not None, if any
                assert isinstance(item, Iterable)
                for excl_item in item:
                    item_vars_name = excl_item['main_argument_name'].lstrip('-').replace('-', '_')
                    if args_dict.get(item_vars_name) is not None:
                        res_params[group][excl_item['parameter_name']] = soft_cast(args_dict.pop(item_vars_name),
                                                                                   excl_item.get('soft_type', None))

                found_values = [res_params[group].get(i['parameter_name']) is not None for i in item]
                if sum(found_values) > 1:
                    raise ValueError('Found multiple values of a mutually exclusive group {}. '
                                     'Potential conflict of --config-file and other arguments. '
                                     'Try deleting the values for conflicting parameters from the config file.'.
                                     format(', '.join(['\'' + i['main_argument_name'] + '\'' for i in item])))
    # if something is left unparsed, it would be otherwise ignored
    if args_dict and not ignore_unused_arguments:
        raise ValueError('Some key(s) in @parsed_args were not used, maybe there were typos in their names: {}'.
                         format(args_dict))

    return res_params

if __name__ == '__main__':
    args = parse_args()
    parameters = create_parameters_from_args(args)
    if parameters['GeneralArguments'].get('verbose'):
        print(util.pretty_string(parameters), file=sys.stderr)
    run_detection(parameters)
