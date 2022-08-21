#!/usr/bin/env python

import os
import sys
import tempfile

from argparse import ArgumentParser
from csv import DictReader
from collections import defaultdict
from itertools import groupby

import numpy as np
import pickle

from sp_tool.arff_helper import ArffHelper
from sp_tool.data_loaders import EM_VALUE_MAPPING_DEFAULT
from sp_tool import util


def maybe_cast_to_float(d):
    """
    Cast dictionary values to float, where possible.
    :param d: dictionary
    :return: dictionary with all values cast to float, where possible
    """
    for key in d:
        try:
            d[key] = float(d[key])
        except ValueError:
            pass
    return d


def load_sampling_parameters(args):
    csv_reader = DictReader(open(args.csv))
    all_events = list(map(maybe_cast_to_float, csv_reader))

    if not args.split_up_attribute:
        event_types = ['FIX', 'SACCADE', 'SP'] + (['NOISE'] if args.noise else [])
    else:
        event_types = ['SACCADE'] + [args.split_up_eye_movement]

    transition_probabilities = defaultdict(dict)
    transition_denom = defaultdict(float)

    plausible_durations_in_samples = defaultdict(list)

    # one_sample_duration_ms = 1e3 / args.sampling_freq

    skipped_samples = 0
    total_samples = 0
    for event in all_events:
        if event['em_type'] in event_types:
            duration = int(event['duration_samples'])
            # duration_ms = int(math.ceil(event['duration_ms'] / one_sample_duration_ms)) + 1
            total_samples += duration

            if args.mode == 'event':
                plausible_durations_in_samples[event['em_type']].append(duration)

            if args.mode == 'sample':
                # in an N-sample event, there are N transitions: N - 1 within the event, 1 more to the next sample
                transition_denom[event['em_type']] += duration - 1

                if event['em_type'] not in transition_probabilities[event['em_type']]:
                    transition_probabilities[event['em_type']][event['em_type']] = 0.0
                transition_probabilities[event['em_type']][event['em_type']] += duration - 1

                # only add the next-sample transition if the next eye movement type is a "valid" one
                if event['successive_em'] in event_types:
                    if event['successive_em'] not in transition_probabilities[event['em_type']]:
                        transition_probabilities[event['em_type']][event['successive_em']] = 0.0
                    transition_probabilities[event['em_type']][event['successive_em']] += 1
                    transition_denom[event['em_type']] += 1
                else:
                    # print 'Unknown next EM type'
                    skipped_samples += 1
            else:  # working with event transitions, one transition per event, and only if the next event is "valid"
                # if the next event is a valid one, include this transition into the transition matrix computation
                if event['successive_em'] in event_types:
                    transition_denom[event['em_type']] += 1

                    if event['successive_em'] not in transition_probabilities[event['em_type']]:
                        transition_probabilities[event['em_type']][event['successive_em']] = 0.0
                    transition_probabilities[event['em_type']][event['successive_em']] += 1
                else:
                    # print 'Unknown next EM type'
                    skipped_samples += 1
        else:
            # print 'Unknown EM type'
            skipped_samples += 1

    # for sample-level randomness, all durations are 1 sample
    if args.mode == 'sample':
        for key in event_types:
            plausible_durations_in_samples[key] = [1]

    # normalise transition probabilities
    for key in event_types:
        for key_next in event_types:
            if key_next not in transition_probabilities[key]:
                transition_probabilities[key][key_next] = 0.0
            transition_probabilities[key][key_next] /= transition_denom[key]

    total_events = sum(transition_denom.values())
    a_priori_prob = {k: v / total_events for k, v in transition_denom.items()}

    print('Skipped', skipped_samples, 'samples/events out, a total of', total_events, 'samples/events analysed')

    return plausible_durations_in_samples, transition_probabilities, a_priori_prob


def compute_mean_std(generator_state):
    if 'duration_mean' not in generator_state or \
                    'duration_std' not in generator_state:
        generator_state['duration_mean'] = {}
        generator_state['duration_std'] = {}
        for key in generator_state['plausible_durations']:
            generator_state['duration_mean'][key] = np.mean(generator_state['plausible_durations'][key])
            generator_state['duration_std'][key] = np.std(generator_state['plausible_durations'][key])
    return generator_state


def generate_next(args, generator_state):
    if generator_state['previous_em'] is None or args.independent:
        event_type = np.random.choice(list(generator_state['a_priori_probs'].keys()),
                                      p=list(generator_state['a_priori_probs'].values()))
    else:
        event_type = np.random.choice(list(generator_state['transition_matrix'][generator_state['previous_em']].keys()),
                                      p=list(generator_state['transition_matrix'][generator_state['previous_em']].values()))

    if not args.simplify:
        event_duration = np.random.choice(generator_state['plausible_durations'][event_type])
    else:
        if 'duration_mean' not in generator_state or \
           'duration_std' not in generator_state:
            # not yet simplified
            generator_state = compute_mean_std(generator_state)
        event_duration = int(round(np.random.normal(loc=generator_state['duration_mean'][event_type],
                                                    scale=generator_state['duration_std'][event_type])))

    generator_state['previous_em'] = event_type

    return {'type': event_type,
            'duration_samples': event_duration}


def preprocess_labels(obj, args):
    if args.split_up_attribute == 'EYE_MOVEMENT_TYPE':
        # the correct attribute already exists, return as-is
        return obj
    obj = util.add_eye_movement_attribute(obj)
    if args.split_up_attribute is not None:
        # if we have the labels already, copy them over to the collect column
        obj['data']['EYE_MOVEMENT_TYPE'] = [EM_VALUE_MAPPING_DEFAULT[x] for x in obj['data'][args.split_up_attribute]]
    return obj


def label_events(args, event_durations, transition_matrix, a_priori_probs):
    np.random.seed(args.random_seed)

    if args.output_folder is None:
        args.output_folder = tempfile.mkdtemp(prefix='random_baseline_')
        print('Creating a temporary folder for the output in "{}"'.format(args.output_folder), file=sys.stderr)

    for root, dirs, files in os.walk(args.input):
        for file in sorted(files):
            if not file.endswith(".arff"):
                continue

            assert root.startswith(args.input)
            subpath = root[len(args.input):].lstrip('/')
            obj = ArffHelper.load(open(os.path.join(root, file)))
            output_fname = os.path.join(args.output_folder, subpath)
            if not os.path.exists(output_fname):
                os.makedirs(output_fname)
            output_fname = os.path.join(output_fname, file)

            obj = preprocess_labels(obj, args)

            generator_state = {
                'previous_em': None,
                'plausible_durations': event_durations,
                'transition_matrix': transition_matrix,
                'a_priori_probs': a_priori_probs
            }

            currently_labelled_i = 0
            while currently_labelled_i < len(obj['data']):
                next_event = generate_next(args, generator_state)
                obj['data']['EYE_MOVEMENT_TYPE'][currently_labelled_i:
                                                 currently_labelled_i + next_event['duration_samples']] = \
                    next_event['type']
                currently_labelled_i += next_event['duration_samples']

            ArffHelper.dump(obj, open(output_fname, 'w')).close()
    pickle.dump(generator_state, open(os.path.join(args.output_folder, 'generator_state_last.pkl'), 'w'))


def split_up_long_events(args, event_durations, transition_matrix, a_priori_probs):
    np.random.seed(args.random_seed)

    if args.output_folder is None:
        args.output_folder = tempfile.mkdtemp(prefix='random_baseline_')
        print('Creating a temporary folder for the output in "{}"'.format(args.output_folder), file=sys.stderr)

    generator_state = {
        'previous_em': None,
        'plausible_durations': event_durations,
        'transition_matrix': transition_matrix,
        'a_priori_probs': a_priori_probs
    }
    generator_state = compute_mean_std(generator_state)

    for root, dirs, files in os.walk(args.input):
        for file in sorted(files):
            if not file.endswith(".arff"):
                continue

            assert root.startswith(args.input)
            subpath = root[len(args.input):].lstrip('/')
            obj = ArffHelper.load(open(os.path.join(root, file)))
            output_fname = os.path.join(args.output_folder, subpath)
            if not os.path.exists(output_fname):
                os.makedirs(output_fname)
            output_fname = os.path.join(output_fname, file)

            obj = preprocess_labels(obj, args)
            final_labels = []  # this will become the split-up labels of @obj in the end

            for em_type, count in [(em, len(list(grp))) for em, grp in groupby(obj['data']['EYE_MOVEMENT_TYPE'])]:
                # another EM type, or a reasonably short event
                if em_type != args.split_up_eye_movement or count <= generator_state['duration_mean'][em_type] + \
                        generator_state['duration_std'][em_type]:
                    final_labels += [em_type] * count
                    continue
                # do not take previous EM info into account
                generator_state['previous_em'] = None
                # a long fixation, will split it up
                currently_labelled_i = 0
                while currently_labelled_i < count:
                    next_event = generate_next(args, generator_state)
                    # limit the generated event's duration with the number of samples that are remaining in this long
                    # fixation
                    next_event['duration_samples'] = min(next_event['duration_samples'], count - currently_labelled_i)

                    final_labels += [next_event['type']] * next_event['duration_samples']
                    currently_labelled_i += next_event['duration_samples']

            obj['data']['EYE_MOVEMENT_TYPE'] = final_labels
            ArffHelper.dump(obj, open(output_fname, 'w')).close()

    pickle.dump(generator_state, open(os.path.join(args.output_folder, 'generator_state_last.pkl'), 'w'))


def parse_args():
    parser = ArgumentParser('Random baseline')
    parser.add_argument('--in', dest='input', default='../GazeCom/gaze_arff/',
                        help='Folder with gaze files to be labelled (folder structure will be re-created).')
    parser.add_argument('--out', dest='output_folder', required=False,
                        help='Where to write the random baseline results.')
    parser.add_argument('--csv', help='The .csv file with events to use as source for random distribution params',
                        default='ground_truth_events.csv')
    parser.add_argument('--mode', choices=['sample', 'event'], default='sample',
                        help='Will generate random samples or events, depending on this parameter')

    parser.add_argument('--independent', action='store_true',
                        help='Only generate events/samples based on their a priori probabilities, '
                             'disregarding transition probabilities')
    parser.add_argument('--noise', action='store_true',
                        help='Keep and generate noise samples/events')

    parser.add_argument('--split-up-eye-movement', '--split-em', default='FIX', required=False, choices=['FIX', 'SP'],
                        help='Split up this eye movement events specifically (fixations by dafault, '
                             'all events above mean + 1xSTD will be split up with saccades at plausible intervals')
    parser.add_argument('--split-up-attribute', '--split', default=None, required=False,
                        help='If this option is used, will attempt to split up fixation episodes (in the labels '
                             'of this attribute) that are over mean + 1xSTD long with the same random baseline. '
                             'Might want to use --no-sp for this.')

    parser.add_argument('--simplify', action='store_true',
                        help='Simplify estimates: approximate distributions as Gaussians')
    parser.add_argument('--random-seed', '--seed', default=0,
                        help='Random seed value')
    return parser.parse_args()


def __main__():
    args = parse_args()

    durations, transitions, a_priori = load_sampling_parameters(args)

    if not args.split_up_attribute:
        label_events(args, event_durations=durations, transition_matrix=transitions, a_priori_probs=a_priori)
    else:
        assert args.mode == 'event', 'Using --split with `--mode sample` is not advised!'
        split_up_long_events(args, event_durations=durations, transition_matrix=transitions, a_priori_probs=a_priori)

if __name__ == '__main__':
    __main__()
