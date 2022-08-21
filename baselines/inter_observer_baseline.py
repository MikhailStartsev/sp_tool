#!/usr/bin/env python

import os
import sys
import tempfile

from argparse import ArgumentParser
import glob
import numpy as np

from collections import Counter

from sp_tool.arff_helper import ArffHelper
from sp_tool import util
from sp_tool.data_loaders import EM_VALUE_MAPPING_DEFAULT


def label_events(args):
    np.random.seed(args.random_seed)

    if args.output_folder is None:
        args.output_folder = tempfile.mkdtemp(prefix='inter_observer_baseline_')
        print('Creating a temporary folder for the output in "{}"'.format(args.output_folder), file=sys.stderr)

    for folder_candidate in sorted(os.listdir(args.input)):
        in_subfolder = os.path.join(args.input, folder_candidate)
        if not os.path.isdir(in_subfolder):
            continue
        out_subfolder = os.path.join(args.output_folder, folder_candidate)
        if not os.path.exists(out_subfolder):
            os.makedirs(out_subfolder)

        all_observers = sorted(glob.glob('{}/*.arff'.format(in_subfolder)))
        matched_observers = all_observers[:]  # copy the list
        while any([x == y for x, y in zip(all_observers, matched_observers)]):
            np.random.shuffle(matched_observers)
        print('For the "{}" stimulus, the following observer recordings are matched:'.\
            format(folder_candidate), file=sys.stderr)
        print(list(zip([os.path.split(x)[1] for x in all_observers],
                                 [os.path.split(x)[1] for x in matched_observers])), file=sys.stderr)

        for target_observer, source_observer in zip(all_observers, matched_observers):
            out_fname = os.path.join(out_subfolder, os.path.split(target_observer)[1])

            obj_target = ArffHelper.load(open(target_observer))
            obj_source = ArffHelper.load(open(source_observer))

            if args.zero_time:
                obj_target['data']['time'] -= obj_target['data']['time'][0]
                obj_source['data']['time'] -= obj_source['data']['time'][0]
            else:
                # Remove the true labels from the target file.
                # Only execute if timestamps are not being modified.
                ArffHelper.remove_column(obj_target, args.expert)
            # add the EYE_MOVEMENT_TYPE column
            obj_target = util.add_eye_movement_attribute(obj_target)

            # external iteration for the @obj_source
            source_i = 0
            # in case there will be no previously assigned label, take the most probable one
            prev_label = Counter(obj_source['data'][args.expert]).most_common(1)[0][0]
            for target_i in range(len(obj_target['data'])):
                assigned_label = None
                if source_i >= len(obj_source['data']):
                    # @source_obj ran out already, just duplicate the label
                    assigned_label = prev_label
                elif obj_source['data']['time'][source_i] > obj_target['data']['time'][target_i] + \
                                args.temporal_tolerance_ms * 1e3:
                    # already too far in the @source_obj, stall its index and assign the @prev_label
                    assigned_label = prev_label
                elif abs(obj_source['data']['time'][source_i] - obj_target['data']['time'][target_i]) <= \
                                args.temporal_tolerance_ms * 1e3:
                    # within the matching distance
                    assigned_label = obj_source['data'][args.expert][source_i]
                    source_i += 1
                else:
                    # we should be too early in the @obj_target, so can possibly scroll through the @obj_source
                    while source_i < len(obj_source['data']) and \
                                    obj_source['data']['time'][source_i] < obj_target['data']['time'][target_i] - \
                                            args.temporal_tolerance_ms * 1e3:
                        source_i += 1
                    # if scrolled too far
                    if source_i >= len(obj_source['data']) or \
                                    obj_source['data']['time'][source_i] > obj_target['data']['time'][target_i] + \
                                            args.temporal_tolerance_ms * 1e3:
                        assigned_label = prev_label
                    else:
                        assert abs(obj_source['data']['time'][source_i] - obj_target['data']['time'][target_i]) <= \
                               args.temporal_tolerance_ms * 1e3
                        assigned_label = obj_source['data'][args.expert][source_i]
                        source_i += 1

                # default value in case of already-categorical labels
                obj_target['data']['EYE_MOVEMENT_TYPE'][target_i] = EM_VALUE_MAPPING_DEFAULT.get(assigned_label,
                                                                                                 assigned_label)
                prev_label = assigned_label

            ArffHelper.dump(obj_target, open(out_fname, 'w')).close()


def parse_args():
    parser = ArgumentParser('Random baseline')
    parser.add_argument('--in', dest='input', default='../GazeCom/ground_truth/',
                        help='Folder with labelled gaze files to be re-labelled (sub-folders corresponding to '
                             'independent stimuli are supposed)')
    parser.add_argument('--out', dest='output_folder', required=False,
                        help='Where to write the random baseline results.')
    parser.add_argument('--expert', default='handlabeller_final',
                        help='Hand-labelling expert column name')
    parser.add_argument('--temporal-tolerance-ms', '--tolerance', default=20, type=float,
                        help='Temporal tolerance of matched samples in milliseconds')
    parser.add_argument('--zero-time', action='store_true',
                        help='Subtract the first time stamp from all the rest '
                             '(in case this is not already done by the data set). In this case will also '
                             'preserve the original labels in the same files, since otherwise it will '
                             'be difficult to run the evaluation.')
    parser.add_argument('--random-seed', '--seed', default=0, type=int,
                        help='Random seed value')
    return parser.parse_args()


def __main__():
    args = parse_args()

    label_events(args)

if __name__ == '__main__':
    __main__()
