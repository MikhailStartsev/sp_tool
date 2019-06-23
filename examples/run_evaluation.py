#!/usr/bin/env python
import sys
import os
import warnings
import glob
import fnmatch
import numpy as np
from argparse import ArgumentParser

# (make sure to first run `python setup.py install --user` in the source directory)
from sp_tool.arff_helper import ArffHelper
from sp_tool.recording_processor import RecordingProcessor
from sp_tool.data_loaders import EM_VALUE_MAPPING_DEFAULT
from sp_tool import evaluate


def find_all_files_with_a_pattern(folder, pattern='*.arff'):
    res = []
    for dirpath, dirnames, filenames in os.walk(folder):
        filenames = fnmatch.filter(filenames, pattern)
        if filenames:
            res += [os.path.join(dirpath, fname) for fname in sorted(filenames)]
    return res


def evaluate_prepared_output(in_folder, movies=None,
                             hand_labelling_folder='../GazeCom/ground_truth/',
                             hand_labelling_expert='handlabeller_final',
                             in_file_wildcard_pattern='*.arff',
                             expert_file_wildcard_pattern='*.arff',
                             algorithm_column=True,
                             return_raw_statistic=False,
                             only_main_eye_movements=False,
                             ignore_gazecom_folder_structure=False,
                             microseconds_in_time_unit=1.0):
    # This extracts the paths to all the labelled files:
    #   - all subdirectories (i.e. all movies)
    #   - all .arff files (i.e. all observers)
    # NB. Using this for both hand-labelling data and the programmatically labelled files assumes
    #   - the same subfolder structure
    #   - the same oder of observers' files in respective subfolders
    # I.e. if the names do not coincide (like renaming from observer names (ex. AAF, YYK, etc.) to 001 ... 050),
    # but *the order is preserved*, this is still okay!
    """
    Evaluate programmatically-produced labels vs hand-labelling data.
    :param in_folder: input folder with subdirectories corresponding to separate clips used during recording
    :param movies: a list of movies to use during evaluation (if None (default) or empty, use all)
    :param hand_labelling_folder: folder with hand-labelling data
    :param hand_labelling_expert: which expert data to use
    :param in_file_wildcard_pattern: wildcard patter if input files (*.arff by default)
    :param expert_file_wildcard_pattern: wildcard patter if ground truth files (*.arff by default)
    :param algorithm_column: the column that contains the output of an eye movement type classification algorithm;
                             by default, will look for EYE_MOVEMENT_TYPE column; if a column (i.e. attribute) name
                             is specified here, will attempt to convert it into a corresponding newly added
                             EYE_MOVEMENT_TYPE column, here only using a default mapping dictionary for numerical
                             labels (0 is 'unknown', 1 corresponding to fixation, 2 -- to saccade,
                             3 -- to smooth pursuit, and 4 - to noise).
                             If you want a more flexible interface, consider using load_multiple_recordings()
                             function of RecordingProcessor programmatically, it supports different conversion
                             options.

                             It is set to True by default, which means that the attribute EYE_MOVEMENT_TYPE is being
                             used. This is suitable, for example, for the .arff files produced by this framework.
    :param return_raw_statistic: whether to return TP/TN/FN/FP instead of F1/precision/recall/etc.
    :param only_main_eye_movements: only evaluate the "main" eye movements: fixations, saccades, and pursuits;
                                    otherwise (by default) will find all eye movement labels and evaluate them
    :param ignore_gazecom_folder_structure: ignore the folder structure of GazeCom (sub-folders for each movie,
                                            files for each subject) and just look for all .arff files in the @in_folder
    :param microseconds_in_time_unit: how many microseconds in a time unit; need this to compute correct event durations
                                      in ms.
    :return: a dictionary with the first-level keys corresponding to the evaluated eye movements; each of those has
             its own sub-dictionary with all the computed metrics
    """
    if not ignore_gazecom_folder_structure:
        if not movies:
            assigned_labels_files = sorted(glob.glob('{}/*/{}'.format(in_folder,
                                                                      in_file_wildcard_pattern)))
            ground_truth_files = sorted(glob.glob('{}/*/{}'.format(hand_labelling_folder,
                                                                   expert_file_wildcard_pattern)))
        else:
            # extract file names from respective directories only
            assigned_labels_files = []
            ground_truth_files = []
            for movie in movies:
                assigned_labels_files += sorted(glob.glob('{}/{}/{}'.format(in_folder, movie, in_file_wildcard_pattern)))
                ground_truth_files += sorted(glob.glob('{}/{}/*.arff'.format(hand_labelling_folder, movie)))
    else:
        ground_truth_files = find_all_files_with_a_pattern(hand_labelling_folder, expert_file_wildcard_pattern)
        assigned_labels_files = find_all_files_with_a_pattern(in_folder, in_file_wildcard_pattern)

    assert len(assigned_labels_files) == len(ground_truth_files), \
        'A different number of input files provided: {} ground truth files and {} labelled files'.format(
            len(ground_truth_files), len(assigned_labels_files)
        )
    assert len(assigned_labels_files) > 0, 'Zero files fit the pattern "{}/*/{}" and "{}/*/{}"'.format(
        in_folder, in_file_wildcard_pattern, hand_labelling_folder, expert_file_wildcard_pattern
    )
    # assigned_labels_objects = [ArffHelper.load(open(fname)) for fname in assigned_labels_files]
    # swapped the ArffHelper-loading of assigned arff files to support numerical labels of eye movements
    #
    # loads all ARFFs, takes a bit of time
    rp = RecordingProcessor()
    assigned_labels_objects = rp.load_multiple_recordings(assigned_labels_files,
                                                          labelled_eye_movement_column_arff=algorithm_column,
                                                          suppress_warnings=True)
    ground_truth_objects = [ArffHelper.load(open(fname)) for fname in ground_truth_files]

    # evaluate for these eye movements (corresponds the possible EYE_MOVEMENT_TYPE labels)
    if only_main_eye_movements:
        em_labels = {'SP', 'FIX', 'SACCADE'}
    else:
        # if we are dealing with categorical attributes, find all labels that are used in the ground truth
        if len(ground_truth_objects) > 0 and \
                        ground_truth_objects[0]['data'][hand_labelling_expert].dtype.type is np.string_:
            all_em_labels = [set(obj['data'][hand_labelling_expert]) for obj in ground_truth_objects]
        else:
            all_em_labels = [set([EM_VALUE_MAPPING_DEFAULT[x] for x in obj['data'][hand_labelling_expert]])
                             for obj in ground_truth_objects]

        em_labels = set().union(*all_em_labels)
    print >> sys.stderr, 'Will evaluate the following labels:', sorted(em_labels)

    # verify that the label sets intersect at all
    all_assigned_labels = [set(obj['data']['EYE_MOVEMENT_TYPE']) for obj in assigned_labels_objects]
    all_assigned_labels = set().union(*all_assigned_labels)
    label_intersection = em_labels.intersection(all_assigned_labels)
    if len(label_intersection) == 0:
        warnings.warn('There is no intersection between evaluated ground truth labels ({}) '
                      'and algorithmically assigned labels ({})!'.format(sorted(em_labels),
                                                                         sorted(all_assigned_labels)))
    elif len(label_intersection) != len(em_labels):
        print >> sys.stderr, 'Intersection between evaluated ground truth labels and the algorithmically ' \
                             'assigned labels is not fully covering the set of evaluated labels: ' \
                             '{} vs {}'.format(sorted(label_intersection), sorted(em_labels))

    res_stats = {}

    # run for each label, and also for all labels at once
    for positive_label in sorted(em_labels) + [None]:
        res_stats[positive_label if positive_label is not None else 'all'] = evaluate.evaluate(
            ground_truth_objects,  # true labels
            assigned_labels_objects,  # assigned labels
            experts=[hand_labelling_expert],  # use just one expert (can alternatively provide a list of names,
                                              # then will use the majority vote
            positive_label=positive_label,  # one evaluation run = one positive label
            return_raw_stats=return_raw_statistic,  # compute F1/precision/... instead of FP/TP/... values
            microseconds_in_time_unit=microseconds_in_time_unit  # for proper durations
        )
    return res_stats


def parse_args():
    parser = ArgumentParser('Evaluate the eye movements detection results by comparing them to hand-labelled data')
    parser.add_argument('--input-folder', '--in', required=True,
                        help='Where to take the .arff data from. The folder is assumed to have separate '
                             'subdirectories for each movie')
    parser.add_argument('--input-file-pattern', '--pattern', required=False, default='*.arff',
                        help='A wildcard-pattern for the files that should be taken as input. '
                             'Can be useful to restrict the script to a part of the files present in the input '
                             'directory, ex. \'*_baseline_*.arff\' vs \'*_updated_*.arff\''
                             'Make sure to put single quotes around it!')
    parser.add_argument('--algorithm-column', '--algorithm', '--alg', default=True,
                        help='The column of the .arff files that presents the algorithmically detected eye movements. '
                             'This facilitates evaluating external labellings. The specified column should '
                             'either contain labels like "FIX", "SP" and "SACCADE", or have numerical values with '
                             '1 corresponding to fixation, 2 -- to saccade and 3 -- to smooth pursuit. '
                             'If you want a more flexible interface, consider using load_multiple_recordings() '
                             'function of RecordingProcessor programmatically, it supports different conversion '
                             'options. '
                             'The default value (True) implies the presence of the EYE_MOVEMENT_TYPE categorical '
                             'attribute. If you have some other input files structure, specify the appropriate '
                             'algorithm column with this argument.')
    parser.add_argument('--hand-labelled', '--hand', required=False,
                        default='../GazeCom/ground_truth/',
                        help='Path to the root folder of the hand-labelled data')
    parser.add_argument('--expert', required=False,
                        default='handlabeller_final',
                        help='Which hand-labelling expert\'s labels to use')
    parser.add_argument('--expert-file-pattern', '--expert-pattern', required=False, default='*.arff',
                        help='A wildcard-pattern for the files that should be taken as the ground truth. '
                             'Can be useful to restrict the script to a part of the files present in the input '
                             'directory, ex. \'*_baseline_*.arff\' vs \'*_updated_*.arff\''
                             'Make sure to put single quotes around it!')
    parser.add_argument('--one-line-output', '--one-line', action='store_true',
                        help='The resulting statistics dictionary will be printed all on one line, instead of the '
                             'default multi-line, but much easier human-readable output. This can be useful for '
                             'storing outputs that correspond to multiple algorithms/parameter combination '
                             'in one file.')
    parser.add_argument('--raw-statistics', '--raw', action='store_true',
                        help='Print out the raw statistics, instead of precision/recall/...')

    parser.add_argument('--all-files', action='store_true',
                        help='Ignore the assumed GazeCom folder structure and just find all files that match the '
                             'pattern (for --input-folder) or are .arff files (for --hand-labelled folder).')
    parser.add_argument('--all-eye-movements', action='store_true',
                        help='Evaluate all eye movements, not just fixations/saccades/pursuits.')
    parser.add_argument('--all', dest='all_allowed', action='store_true',
                        help='Equivalent to setting --all-files and --all-eye-movements together.')

    parser.add_argument('--microseconds-in-time-unit', '--microsec', default=1.0, type=float,
                        help='How many microseconds in the time unit of the recordings (the "time" column)')

    args = parser.parse_args()

    if args.all_allowed:
        args.all_files = True
        args.all_eye_movements = True

    print >> sys.stderr, 'For event duration computation assuming {} microseconds in one time unit. Change ' \
                         'with --microsec, if necessary.'.format(args.microseconds_in_time_unit)

    return args

if __name__ == '__main__':
    args = parse_args()
    stats = evaluate_prepared_output(in_folder=args.input_folder,
                                     hand_labelling_folder=args.hand_labelled,
                                     hand_labelling_expert=args.expert,
                                     in_file_wildcard_pattern=args.input_file_pattern,
                                     expert_file_wildcard_pattern=args.expert_file_pattern,
                                     algorithm_column=args.algorithm_column,
                                     return_raw_statistic=args.raw_statistics,
                                     ignore_gazecom_folder_structure=args.all_files,
                                     only_main_eye_movements=not args.all_eye_movements,
                                     microseconds_in_time_unit=args.microseconds_in_time_unit)
    if args.one_line_output:
        print stats
    else:
        import json
        print json.dumps(stats, sort_keys=True, separators=[',', ': '], indent=4)
