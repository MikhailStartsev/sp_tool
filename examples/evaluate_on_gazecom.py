#!/usr/bin/env python
from argparse import ArgumentParser
import glob

# (make sure to first run `python setup.py install --user` in the source directory)
from sp_tool.arff_helper import ArffHelper
from sp_tool.recording_processor import RecordingProcessor
from sp_tool import evaluate


def evaluate_prepared_output(in_folder, movies=None,
                             hand_labelling_folder='../GazeCom/ground_truth/',
                             hand_labelling_expert='handlabeller_final',
                             in_file_wildcard_pattern='*.arff',
                             algorithm_column=True,
                             return_raw_statistic=False):
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
    :return:
    """
    if not movies:
        assigned_labels_files = sorted(glob.glob('{}/*/{}'.format(in_folder, in_file_wildcard_pattern)))
        ground_truth_files = sorted(glob.glob('{}/*/*.arff'.format(hand_labelling_folder)))
    else:
        # extract file names from respective directories only
        assigned_labels_files = []
        ground_truth_files = []
        for movie in movies:
            assigned_labels_files += sorted(glob.glob('{}/{}/{}'.format(in_folder, movie, in_file_wildcard_pattern)))
            ground_truth_files += sorted(glob.glob('{}/{}/*.arff'.format(hand_labelling_folder, movie)))

    assert len(assigned_labels_files) == len(ground_truth_files), \
        'A different number of input files provided: {} ground truth files and {} labelled files'.format(
            len(ground_truth_files), len(assigned_labels_files)
        )

    # assigned_labels_objects = [ArffHelper.load(open(fname)) for fname in assigned_labels_files]
    # swapped the ArffHelper-loading of assigned arff files to support numerical labels of eye movements
    # loads all ARFFs, takes a bit of time
    rp = RecordingProcessor()
    assigned_labels_objects = rp.load_multiple_recordings(assigned_labels_files,
                                                          labelled_eye_movement_column_arff=algorithm_column)
    ground_truth_objects = [ArffHelper.load(open(fname)) for fname in ground_truth_files]

    # evaluate for these eye movements (corresponds the possible EYE_MOVEMENT_TYPE labels)
    em_labels = ['SP', 'FIX', 'SACCADE']
    res_stats = {}

    for positive_label in em_labels:
        res_stats[positive_label] = evaluate.evaluate(
            ground_truth_objects,  # true labels
            assigned_labels_objects,  # assigned labels
            experts=[hand_labelling_expert],  # use just one expert (can alternatively provide a list of names,
                                              # then will use the majority vote
            positive_label=positive_label,  # one evaluation run = one positive label
            return_raw_stats=return_raw_statistic  # compute F1/precision/... instead of FP/TP/... values
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
    parser.add_argument('--pretty', '--pretty-print', action='store_true',
                        help='Use pretty-printing of the resulting dictionary (results in a multi-line, '
                             'but human-readable output)')
    parser.add_argument('--raw-statistics', '--raw', action='store_true',
                        help='Print out the raw statistics, instead of precision/recall/...')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    stats = evaluate_prepared_output(in_folder=args.input_folder,
                                     hand_labelling_folder=args.hand_labelled,
                                     hand_labelling_expert=args.expert,
                                     in_file_wildcard_pattern=args.input_file_pattern,
                                     algorithm_column=args.algorithm_column,
                                     return_raw_statistic=args.raw_statistics)
    if not args.pretty:
        print stats
    else:
        import json
        print json.dumps(stats, sort_keys=True, separators=[',', ': '], indent=4)
