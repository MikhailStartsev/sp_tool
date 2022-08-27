#!/usr/bin/env python
# -*- coding: utf8 -*-
import string
import numpy as np
import itertools
from collections import Counter, defaultdict
import sys
import warnings
import copy

from sklearn.metrics import cohen_kappa_score
from jellyfish import levenshtein_distance

from sp_tool.data_loaders import EM_VALUE_MAPPING_DEFAULT
from sp_tool import util


class Event(object):
    def __init__(self, event_type, start, end, duration):
        self.type = event_type
        self.start = start
        self.end = end
        self.duration = duration

# Ground truth is usually stored in ARFF files with a separate column for each hand-labelling expert (if multiple are
# present). Each hand-labelling expert column contains numerical data with the following correspondence to eye movement
# types:
CORRESPONDENCE_TO_HAND_LABELLING_VALUES = {value: key for key, value in EM_VALUE_MAPPING_DEFAULT.items()}


def get_majority_vote_efficient(obj, experts, positive_label):
    """
    Get majority vote of labels among different experts, in a more efficient way through numpy functions.

    :param obj: arff object
    :param experts: list of experts (i.e. column or attribute names that are considered in the vote).
    :param positive_label: the label to be evaluated (all values in @experts columns are treated as "yes"
                           (i.e. equals to the positive label) or "no" (i.e. does not)).
    :return: majority vote of label as binary values, with 1 meaning that the majority agrees to assign
             the @positive_label to this row of @obj['data'], and 0 -- that it does not.

    Example:
    get_majority_vote_efficient(arff_object, ['expert1', 'expert2'], 'SP')

    """
    assert len(experts) >= 1
    # determine the type of labels
    label_dtype = obj['data'][experts[0]].dtype
    if len(experts) == 1:
        # just one expert, he always agrees with himself
        if label_dtype.type is np.string_:
            # ground truth labels are already strings, no need for conversion
            return obj['data'][experts[0]] == positive_label
        else:  # ground truth labels are not strings, convert using the standard dictionary
            return obj['data'][experts[0]] == CORRESPONDENCE_TO_HAND_LABELLING_VALUES[positive_label]
    else:
        hand_labellings = obj['data'][experts]
        hand_labellings_list = hand_labellings.view((label_dtype, len(hand_labellings.dtype.names)))
        thd = len(experts) / 2.0  # at least 50% of experts should agree
        if label_dtype.type is np.string_:
            majority_vote = ((hand_labellings_list == positive_label).sum(axis=1) >= thd).astype(int)
        else:
            majority_vote = ((hand_labellings_list ==
                              CORRESPONDENCE_TO_HAND_LABELLING_VALUES[positive_label]).sum(axis=1) >= thd).astype(int)
        return majority_vote


def get_majority_vote(obj, experts, exclude_values=None):
    """
    Get majority vote of labels among different experts, keeping all the labels as possible.

    :param obj: arff objects
    :param experts: list of experts (i.e. column or attribute names that are considered in the vote).
    :param exclude_values: a list of values that should be excluded from the vote (ex. ['UNKNOWN'], or [0]);
                           if not a list, will be converted to a list with 1 element;
                           if all the experts vote for one of these values, they will be taken into account.
    :return: majority vote of label in form of array of label number.

    Example:
    get_majority_vote(arff_object1, ['expert1', 'expert2'])

    """
    # determine the type of labels
    label_dtype = obj['data'][experts[0]].dtype

    if len(experts) == 1:
        return obj['data'][experts].astype(label_dtype if label_dtype.type is np.string_ else int)
    else:
        if exclude_values is not None:
            if not isinstance(exclude_values, list):
                exclude_values = [exclude_values]
        else:
            exclude_values = []
        exclude_values = set(exclude_values)

        majority_vote = []
        hand_labellings = obj['data'][experts].tolist()
        for i in range(len(hand_labellings)):
            # max number of occurrences wins
            candidates_set = set(hand_labellings[i]).difference(exclude_values)
            if not candidates_set:
                candidates_set = set(hand_labellings[i])
            majority_vote.append(max(candidates_set, key=hand_labellings[i].count))
        majority_vote = np.asarray(majority_vote, dtype=label_dtype if label_dtype.type is np.string_ else np.int)

        return majority_vote


def compute_statistics(raw_stats):
    """
    Convert raw statistics in @raw_stats to some more usable format
    :param raw_stats: a dictionary with keys "TP", "TN", "FP", "FN" for true positives count, true negatives,
                      false positives and false negatives, respectively.
    :return: a dictionary with keys "precision", "recall", "F1", "FPR" (for False Positive Rate), "accuracy".
    """

    # for Kappa computation
    if (raw_stats['TP'] + raw_stats['FP'] + raw_stats['TN'] + raw_stats['FN']) != 0:
        p_chance = float((raw_stats['TP'] + raw_stats['FP']) * (raw_stats['TP'] + raw_stats['FN']) +
                         (raw_stats['FP'] + raw_stats['TN']) * (raw_stats['FN'] + raw_stats['TN'])) / \
                   ((raw_stats['TP'] + raw_stats['FP'] + raw_stats['TN'] + raw_stats['FN']) ** 2)
        p_observed = float(raw_stats['TP'] + raw_stats['TN']) / (raw_stats['TP'] +
                                                                 raw_stats['FP'] +
                                                                 raw_stats['TN'] +
                                                                 raw_stats['FN'])
    else:
        p_chance = 0.0
        p_observed = 0.0

    res = {
        'precision': (float(raw_stats['TP']) / (raw_stats['TP'] + raw_stats['FP']))
        if (raw_stats['TP'] + raw_stats['FP']) != 0 else 0.0,

        'recall': (float(raw_stats['TP']) / (raw_stats['TP'] + raw_stats['FN']))
        if (raw_stats['TP'] + raw_stats['FN']) != 0 else 0.0,

        'F1': (2 * float(raw_stats['TP']) / (2 * raw_stats['TP'] + raw_stats['FP'] + raw_stats['FN']))
        if (2 * raw_stats['TP'] + raw_stats['FP'] + raw_stats['FN']) != 0 else 0.0,

        'FPR': (float(raw_stats['FP']) / (raw_stats['FP'] + raw_stats['TN']))
        if (raw_stats['FP'] + raw_stats['TN']) != 0 else 0.0,

        'accuracy':
            (float(raw_stats['TP'] + raw_stats['TN']) / (raw_stats['TP'] +
                                                         raw_stats['FP'] +
                                                         raw_stats['TN'] +
                                                         raw_stats['FN']))
        if (raw_stats['TP'] + raw_stats['FP'] + raw_stats['TN'] + raw_stats['FN']) != 0 else 0.0,

        'sensitivity': (float(raw_stats['TP']) / (raw_stats['TP'] + raw_stats['FN']))
        if (raw_stats['TP'] + raw_stats['FN']) != 0 else 0.0,

        'specificity': (float(raw_stats['TN']) / (raw_stats['TN'] + raw_stats['FP']))
        if (raw_stats['TN'] + raw_stats['FP']) != 0 else 0.0,

        'kappa': ((p_observed - p_chance) / (1 - p_chance)) if p_chance != 1.0 else 0.0
    }
    return res


def extract_events(labels, type_mapping_dict=None):
    events = []

    current_i = 0
    for grp_key, grp_val in itertools.groupby(labels):
        event_len = len(list(grp_val))
        event_type = grp_key
        if type_mapping_dict is not None:
            if event_type in type_mapping_dict:
                event_type = type_mapping_dict[event_type]
            elif not isinstance(event_type, str):
                warnings.warn('A non-string label "{}" not found in the @type_maping_dict, keeping the label as-is.'.
                              format(event_type))
        events.append(Event(event_type=event_type, start=current_i, end=current_i + event_len, duration=event_len))
        current_i += event_len
    return events


def check_event_intersection(event_a, event_b,
                             intersection_over_union_threshold=0.0,
                             return_iou=False):
    later_start = max(event_a.start, event_b.start)
    earlier_end = min(event_a.end, event_b.end)
    intersection_duration = earlier_end - later_start

    if intersection_duration <= 0:
        if not return_iou:
            return False
        else:
            return False, 0.0
    elif not return_iou:
        #  intersection non-zero, no need to compute IoU
        return True

    earlier_start = min(event_a.start, event_b.start)
    later_end = max(event_a.end, event_b.end)
    union_duration = later_end - earlier_start

    iou = float(intersection_duration) / union_duration
    if not return_iou:
        return iou >= intersection_over_union_threshold
    else:
        return iou >= intersection_over_union_threshold, iou


def evaluate_normalised_Levenshtein_dist(true_labels_list,
                                         assigned_labels_list,
                                         experts, positive_label='SP',
                                         return_raw_stats=False,
                                         verbose=False):
    """
    Sample- and episode-level normalised Levenshtein distance (used by [1] as EER and SER). Will also compute the
    sample error rate (Hamming distance).

    :param true_labels_list: list of arff objects produced with hand-labelling tool [2].
    :param assigned_labels_list: list of arff objects produced with this tool (or loaded via RecordingProcessor).
    :param experts: list of experts (for our data, one expert was the tie-corrector, so normally a list of one element
                    should be used, ['handlabeller_final']).
    :param positive_label: the positive label to be evaluated, usually 'SP'/'FIX'/'SACCADE' or None;
                           if None, will evaluate for all labels combined.
    :param return_raw_stats: whether to return raw statistics (lists of values) instead of averages
    :param verbose: output runtime (debug) information
    :return: evaluation results in a dictionary form

    [1] https://link.springer.com/article/10.3758%2Fs13428-018-1133-5
    [2] http://ieeexplore.ieee.org/abstract/document/7851169/
    """
    stats = {
        'sample': [],
        'episode': [],
        'error_rate': {'nom': 0.0, 'denom': 0.0}
    }

    # recover the proper names of the events from hand-labelled data with the default scheme
    mapping_labels_to_names = EM_VALUE_MAPPING_DEFAULT

    for ground_truth, assigned_labels in zip(true_labels_list, assigned_labels_list):
        # check that the t-x-y data has all at least similar values
        assert np.allclose(ground_truth['data']['time'], assigned_labels['data']['time'])
        assert np.allclose(ground_truth['data']['x'], assigned_labels['data']['x'])
        assert np.allclose(ground_truth['data']['y'], assigned_labels['data']['y'])

        ground_truth_labels = get_majority_vote(ground_truth, experts)
        # convert to label names
        ground_truth_labels = [mapping_labels_to_names.get(val, val) for val in ground_truth_labels]

        # string.printable without the string.whitespace characters
        characters_to_encode_labels = string.digits + string.ascii_letters + string.punctuation
        all_unique_labels = sorted(set(ground_truth_labels).union(set(assigned_labels['data']['EYE_MOVEMENT_TYPE'])))
        assert len(all_unique_labels) <= len(characters_to_encode_labels), 'Too many ({}) possible labels, cannot ' \
                                                                           'encode as single symbols more than {}. ' \
                                                                           'Consider using fewer labels or running ' \
                                                                           'this evaluation for each label ' \
                                                                           'independently by passing a ' \
                                                                           '@positive_label parameter.'.\
            format(len(all_unique_labels), len(characters_to_encode_labels))
        all_unique_labels_mapping = {val: characters_to_encode_labels[i] for i, val in enumerate(all_unique_labels)}
        if positive_label is not None:
            for key in all_unique_labels_mapping:
                if key != positive_label:
                    all_unique_labels_mapping[key] = '0'
                else:
                    all_unique_labels_mapping[key] = '1'

        if verbose:
            print('For the positive label of {}, using the following mapping: {}'.format(positive_label,
                                                                                     all_unique_labels_mapping))

        # Sample-level distance
        symbol_sequence_true = ''.join([all_unique_labels_mapping[x]
                                        for x in ground_truth_labels])
        symbol_sequence_assigned = ''.join([all_unique_labels_mapping[x]
                                            for x in assigned_labels['data']['EYE_MOVEMENT_TYPE']])
        stats['sample'].append(float(levenshtein_distance(symbol_sequence_assigned, symbol_sequence_true)) /
                               max(len(symbol_sequence_assigned), len(symbol_sequence_true)))

        stats['error_rate']['nom'] += (np.array(list(symbol_sequence_true)) !=
                                       np.array(list(symbol_sequence_assigned))).sum()
        stats['error_rate']['denom'] += len(symbol_sequence_assigned)

        # Event-level distance
        ground_truth_events = extract_events(ground_truth_labels)
        assigned_events = extract_events(assigned_labels['data']['EYE_MOVEMENT_TYPE'])

        symbol_sequence_true = ''.join([all_unique_labels_mapping[x.type] for x in ground_truth_events])
        symbol_sequence_assigned = ''.join([all_unique_labels_mapping[x.type] for x in assigned_events])

        stats['episode'].append(float(levenshtein_distance(symbol_sequence_assigned, symbol_sequence_true)) /
                                max(len(symbol_sequence_assigned), len(symbol_sequence_true)))

    if return_raw_stats:
        return stats

    for key in stats:
        if key != 'error_rate':
            stats[key] = np.mean(stats[key])
        else:
            stats[key] = stats[key]['nom'] / (stats[key]['denom'] if stats[key]['denom'] != 0 else 1.0)

    return stats


def evaluate_basic_statistics(true_labels_list,
                              assigned_labels_list,
                              experts, positive_label='SP',
                              return_raw_stats=False,
                              microseconds_in_time_unit=1.0):
    """
    Event-level basic statistics: number of events, average duration, average amplitude.
    Alternatively, sample-level percentages of samples, if @positive_label is None.

    :param true_labels_list: list of arff objects produced with hand-labelling tool [1].
    :param assigned_labels_list: list of arff objects produced with this tool (or loaded via RecordingProcessor).
    :param experts: list of experts (for our data, one expert was the tie-corrector, so normally a list of one element
                    should be used, ['handlabeller_final']).
    :param positive_label: the positive abel to be evaluated (normally would be 'SP'/'FIX'/'SACCADE');
                           if None, will compute proportions of samples of all eye movement types
    :param return_raw_stats: whether to return raw statistics (lists of values) instead of averages
    :param microseconds_in_time_unit: how many microseconds in one unit of the 'time' attribute (1 for GazeCom);
                                      if not 1 and not provided, treat "duration_ms" as a non-normalised duration
                                      measurement
    :return: evaluation results in a dictionary form

    [1] http://ieeexplore.ieee.org/abstract/document/7851169/
    """
    amplitude_key = 'amplitude_deg'
    try:
        _ = util.calculate_ppd(true_labels_list[0], skip_consistency_check=True)
    except:
        amplitude_key = 'amplitude_px'

    # different statistics are computed depending on the @positive_label
    if positive_label is not None:
        stats_to_initialise = [('count', 0.0),
                               ('duration_ms', []),
                               (amplitude_key, [])]
    else:
        stats_to_initialise = [('samples_amount', defaultdict(float))]
        # just for validation
        total_samples = {'true': 0, 'detected': 0}

    stats = {
        'true': dict(copy.deepcopy(stats_to_initialise)),  # have to copy, otherwise the dictionaries will be identical
        'detected': dict(copy.deepcopy(stats_to_initialise))
    }

    # recover the proper names of the events from hand-labelled data with the default scheme
    mapping_labels_to_names = EM_VALUE_MAPPING_DEFAULT

    for ground_truth, assigned_labels in zip(true_labels_list, assigned_labels_list):
        # check that the t-x-y data has all at least similar values
        assert np.allclose(ground_truth['data']['time'], assigned_labels['data']['time'])
        assert np.allclose(ground_truth['data']['x'], assigned_labels['data']['x'])
        assert np.allclose(ground_truth['data']['y'], assigned_labels['data']['y'])

        ground_truth_labels = get_majority_vote(ground_truth, experts)

        if positive_label is not None:
            # skip last to avoid border effect
            ground_truth_events = extract_events(ground_truth_labels, type_mapping_dict=mapping_labels_to_names)[:-1]
            assigned_events = extract_events(assigned_labels['data']['EYE_MOVEMENT_TYPE'])[:-1]

            def filter_lambda(x):
                return x.type == positive_label
            ground_truth_events = list(filter(filter_lambda, ground_truth_events))
            assigned_events = list(filter(filter_lambda, assigned_events))

            for stats_key, evaluated_events in zip(['true', 'detected'],
                                                   [ground_truth_events, assigned_events]):
                stats[stats_key]['count'] += len(evaluated_events)

                stats[stats_key]['duration_ms'] += [(ground_truth['data']['time'][e.end] -
                                                     ground_truth['data']['time'][e.start]) * microseconds_in_time_unit
                                                    * 1e-3  # convert time units to microsec, then to ms
                                                    for e in evaluated_events]
                if amplitude_key.endswith('deg'):
                    ppd = util.calculate_ppd(ground_truth, skip_consistency_check=True)
                else:
                    ppd = 1.0

                stats[stats_key][amplitude_key] += [np.linalg.norm([(ground_truth['data'][coord][e.end] -
                                                                     ground_truth['data'][coord][e.start]) / ppd
                                                                    for coord in ['x', 'y']])
                                                    for e in evaluated_events]
        else:
            # convert to label names
            ground_truth_labels = np.array([mapping_labels_to_names.get(val, val) for val in ground_truth_labels])
            total_samples['true'] += len(ground_truth_labels)
            total_samples['detected'] += len(assigned_labels['data'])

            for label in set(ground_truth_labels):
                stats['true']['samples_amount'][label] += (ground_truth_labels == label).sum()
            for label in set(assigned_labels['data']['EYE_MOVEMENT_TYPE']):
                stats['detected']['samples_amount'][label] += (assigned_labels['data']['EYE_MOVEMENT_TYPE'] ==
                                                               label).sum()

    if positive_label is None:
        # validate sample counts
        for key in stats:
            assert sum(stats[key]['samples_amount'].values()) == total_samples[key]

    if return_raw_stats:
        return stats

    if positive_label is not None:
        for key in stats:
            for normalised_key in [amplitude_key, 'duration_ms']:
                stats[key][normalised_key] = np.mean(stats[key][normalised_key])
    else:
        for key in stats:
            denom = sum(stats[key]['samples_amount'].values())
            stats[key]['samples_amount'] = {k: v / denom for k, v in stats[key]['samples_amount'].items()}

    return stats


def evaluate_episodes_adjusted_Cohens_kappa(true_labels_list,
                                            assigned_labels_list,
                                            experts, positive_label='SP',
                                            only_match_positive_events=True,
                                            intersection_over_union_threshold=0.0,
                                            random_seed=0,
                                            return_raw_stats=True,
                                            num_runs=1,
                                            verbose=False):
    """
    The corrected version of the event-level Cohen's kappa scores [1] evaluation of the labelling result
    (algorithm output) in @assigned_labels_list with hand-labelling expert's labels in @ground_truth_list.
    Differently from [2], IoU is used to pick the best match instead of simple intersection.
    Also added an option of limiting event matches to those with good IoU scores.

    Cohen's kappa essentially compares the observed level of accuracy (event-level in this case) to the chance
    level of agreement. We here modify the chance-level performance to eliminate bias against short events by
    estimating the agreement of the randomly-shuffled sequence of events, and NOT the event labels like in [2].

    Will match event by the largest IoU, create two list of labels (one for @true_labels_list,
    one for @assigned_labels_list) that consist of three blocks:
      (1) matched event labels
      (2) missed event labels for the @true_labels_list-associated list, UNKNOWN-labels for the
          @assigned_labels_list-associated list
      (3) false alarm labels for the @assigned_labels_list-associated list, same number of UNKNOWN labels for the
          @true_labels_list-associated list.
    Will then compute standard Cohen's kappa on these.

    Modifications over [2]:
        - largest IoU used instead of simple intersection
        - possibility to start "matching" events only when a certain IoU threshold is exceeded.

    :param true_labels_list: list of arff objects produced with hand-labelling tool [3].
    :param assigned_labels_list: list of arff objects produced with this tool (or loaded via RecordingProcessor).
    :param experts: list of experts (for our data, one expert was the tie-corrector, so normally a list of one element
                    should be used, ['handlabeller_final']).
    :param positive_label: the positive abel to be evaluated (normally would be 'SP'/'FIX'/'SACCADE'); can be None,
                           in which case will run the kappa computation with all labels present.
    :param only_match_positive_events: if True (default), will only consider positive-label events for computing
                                       the observed and chance agreement; ignored for @positive_label=None
    :param intersection_over_union_threshold: (has to be a floating point number in the range of [0; 1]) only count
                                              a "hit", if the IoU is no smaller than this threshold
    :param random_seed: seed to the random shuffling
    :param return_raw_stats: return a list of kappas, before averaging
    :param num_runs: how many times to run the random event re-shuffling
    :param verbose: output runtime (debug) information
    :return: evaluation results in a dictionary form

    [1] https://dl.acm.org/citation.cfm?id=3319836 - "A novel gaze event detection metric that is not fooled by
    gaze-independent baselines", Startsev et al. 2019
    [2] https://link.springer.com/article/10.3758%2Fs13428-018-1133-5
    [3] http://ieeexplore.ieee.org/abstract/document/7851169/
    """
    if positive_label is None:
        only_match_positive_events = False

    rand = np.random.RandomState(seed=random_seed)
    res = {'kappa': []}

    # recover the proper names of the events from hand-labelled data with the default scheme
    mapping_labels_to_names = EM_VALUE_MAPPING_DEFAULT

    for ground_truth, assigned_labels in zip(true_labels_list, assigned_labels_list):
        # check that the t-x-y data has all at least similar values
        assert np.allclose(ground_truth['data']['time'], assigned_labels['data']['time'])
        assert np.allclose(ground_truth['data']['x'], assigned_labels['data']['x'])
        assert np.allclose(ground_truth['data']['y'], assigned_labels['data']['y'])

        ground_truth_labels = get_majority_vote(ground_truth, experts)
        ground_truth_events = extract_events(ground_truth_labels, type_mapping_dict=mapping_labels_to_names)
        assigned_events = extract_events(assigned_labels['data']['EYE_MOVEMENT_TYPE'])

        if positive_label is not None:
            for e in assigned_events:
                if e.type != positive_label:
                    e.type = '_WRONG_LABEL'
            for e in ground_truth_events:
                if e.type != positive_label:
                    e.type = '_WRONG_LABEL'

        assigned_events_shuffled_many = []
        for _ in range(num_runs):
            assigned_events_shuffled = copy.deepcopy(assigned_events)
            rand.shuffle(assigned_events_shuffled)
            current_i = 0
            for e in assigned_events_shuffled:
                e.start = current_i
                e.end = e.start + e.duration
                current_i += e.duration
            assigned_events_shuffled_many.append(assigned_events_shuffled)

        accuracies = {'observed': [], 'chance': []}

        for key, evaluated_events in zip(['observed'] + ['chance'] * len(assigned_events_shuffled_many),
                                         [assigned_events] + assigned_events_shuffled_many):
            acc_nom = 0.0
            acc_denom = 0.0

            assigned_event_i = 0
            for ground_truth_event in ground_truth_events:
                # find the intersecting assigned events
                # skip through the events that end before the current ground truth one
                while assigned_event_i < len(evaluated_events) and \
                                evaluated_events[assigned_event_i].end <= ground_truth_event.start:
                    # detected event that missed
                    if not only_match_positive_events or evaluated_events[assigned_event_i].type == positive_label:
                        acc_denom += 1
                    assigned_event_i += 1

                hit_event_i = None
                hit_iou = 0.0

                candidate_event_i = assigned_event_i
                # while the events keep (potentially) intersecting, keep iterating and checking the intersection
                while candidate_event_i < len(evaluated_events) and \
                      evaluated_events[candidate_event_i].start < ground_truth_event.end:
                    intersection_flag, iou = check_event_intersection(ground_truth_event,
                                                                      evaluated_events[candidate_event_i],
                                                                      intersection_over_union_threshold=
                                                                      intersection_over_union_threshold,
                                                                      return_iou=True)
                    if intersection_flag:
                        # found the valid intersection of events, but is it the highest IoU?
                        if iou > hit_iou:
                            hit_event_i = candidate_event_i
                            hit_iou = iou
                    else:
                        # intersection criteria are not fulfilled, but maybe there are better-intersected events ahead
                        pass
                    candidate_event_i += 1

                if hit_event_i is None:
                    # no match found -> Miss
                    if not only_match_positive_events or ground_truth_event.type == positive_label:
                        acc_denom += 1
                else:
                    # Found some match, set all events between @assigned_event_i and @hit_event_i as false alarms
                    for candidate_event_i in range(assigned_event_i, hit_event_i):
                        if not only_match_positive_events or evaluated_events[candidate_event_i].type == positive_label:
                            acc_denom += 1
                    # matched a pair = 2 events are "accounted for"
                    if not only_match_positive_events:
                        acc_denom += 2
                    else:
                        if ground_truth_event.type == positive_label:
                            acc_denom += 1
                        if evaluated_events[hit_event_i].type == positive_label:
                            acc_denom += 1
                    # only if the even types match, the accuracy nominator should be increased
                    if ground_truth_event.type == evaluated_events[hit_event_i].type:
                        # if matching only the positive type or the label matches either way
                        if not only_match_positive_events or ground_truth_event.type == positive_label:
                            acc_nom += 2

                    # Move @assigned_event_i to @hit_i + 1 and start further processing there
                    assigned_event_i = hit_event_i + 1

            # went through all the ground truth events, let's see whether any detected events remain (all False Alarms)
            for candidate_event_i in range(assigned_event_i, len(evaluated_events)):
                if not only_match_positive_events or evaluated_events[candidate_event_i].type == positive_label:
                    acc_denom += 1

            if not only_match_positive_events:
                assert len(ground_truth_events) + len(evaluated_events) == acc_denom, \
                    'Different number of events matched + not matched compared to the total number of events: {} events ' \
                    'in two lists in total, {} -- after (not) matching'.format(len(ground_truth_events) + len(evaluated_events),
                                                                               acc_denom)
            else:
                num_pos_events = len([e for e in ground_truth_events + evaluated_events if e.type == positive_label])
                assert num_pos_events == acc_denom, 'Found {} events in the accuracy denominator instead of expected {}'\
                    .format(acc_denom, num_pos_events)
            accuracies[key].append((acc_nom / acc_denom) if acc_denom > 0 else 0.0)

        for key in accuracies:
            accuracies[key] = np.mean(accuracies[key])
        if accuracies['chance'] != 1:
            res['kappa'].append((accuracies['observed'] - accuracies['chance']) / (1 - accuracies['chance']))
        else:
            if accuracies['observed'] == 1.0:
                res['kappa'].append(0.0)
            else:
                res['kappa'].append(-1.0)

    if return_raw_stats:
        return res

    res['kappa'] = np.nanmean(res['kappa'])
    return res


def evaluate_episodes_as_Zemblys_et_al(true_labels_list,
                                       assigned_labels_list,
                                       experts, positive_label='SP',
                                       intersection_over_union_threshold=0.0,
                                       verbose=False):
    """
    Event-level Cohen's kappa scores [1] evaluation of the labelling result (algorithm output) in @assigned_labels_list
    with hand-labelling expert's labels in @ground_truth_list. Different from [1], IoU is used to pick the best
    match instead of simple intersection. Also added an option of limiting event matches to those with good IoU scores.

    Will match event by the largest IoU, create two list of labels (one for @true_labels_list,
    one for @assigned_labels_list) that consist of three blocks:
      (1) matched event labels
      (2) missed event labels for the @true_labels_list-associated list, UNKNOWN-labels for the
          @assigned_labels_list-associated list
      (3) false alarm labels for the @assigned_labels_list-associated list, same number of UNKNOWN labels for the
          @true_labels_list-associated list.
    Will then compute standard Cohen's kappa on these.

    Modifications over [1]:
        - largest IoU used instead of simple intersection
        - possibility to start "matching" events only when a certain IoU threshold is exceeded.

    :param true_labels_list: list of arff objects produced with hand-labelling tool [2].
    :param assigned_labels_list: list of arff objects produced with this tool (or loaded via RecordingProcessor).
    :param experts: list of experts (for our data, one expert was the tie-corrector, so normally a list of one element
                    should be used, ['handlabeller_final']).
    :param positive_label: the positive abel to be evaluated (normally would be 'SP'/'FIX'/'SACCADE'); can be None,
                           in which case will run the kappa computation with all labels present.
    :param intersection_over_union_threshold: (has to be a floating point number in the range of [0; 1]) only count
                                              a "hit", if the IoU is no smaller than this threshold
    :param verbose: output runtime (debug) information
    :return: evaluation results in a dictionary form

    [1] https://link.springer.com/article/10.3758%2Fs13428-018-1133-5
    [2] http://ieeexplore.ieee.org/abstract/document/7851169/
    """
    true_labels_events = []
    assigned_labels_events = []
    total_events = {'true': 0, 'assigned': 0}

    # recover the proper names of the events from hand-labelled data with the default scheme
    mapping_labels_to_names = EM_VALUE_MAPPING_DEFAULT

    for ground_truth, assigned_labels in zip(true_labels_list, assigned_labels_list):
        # check that the t-x-y data has all at least similar values
        assert np.allclose(ground_truth['data']['time'], assigned_labels['data']['time'])
        assert np.allclose(ground_truth['data']['x'], assigned_labels['data']['x'])
        assert np.allclose(ground_truth['data']['y'], assigned_labels['data']['y'])

        # matched events will go directly to @true_labels_events and @assigned_labels_events
        # unmatched events will go in either of these buffers, depending on the source of it
        missed_labels_buffer = []
        false_alarm_labels_buffer = []

        ground_truth_labels = get_majority_vote(ground_truth, experts)
        ground_truth_events = extract_events(ground_truth_labels, type_mapping_dict=mapping_labels_to_names)
        assigned_events = extract_events(assigned_labels['data']['EYE_MOVEMENT_TYPE'])

        total_events['true'] += len(ground_truth_events)
        total_events['assigned'] += len(assigned_events)

        assigned_event_i = 0
        local_matched_events_count = 0
        for ground_truth_event in ground_truth_events:
            # find the intersecting assigned events
            # skip through the events that end before the current ground truth one
            while assigned_event_i < len(assigned_events) and \
                  assigned_events[assigned_event_i].end <= ground_truth_event.start:

                false_alarm_labels_buffer.append(assigned_events[assigned_event_i].type)
                assigned_event_i += 1

            hit_event_i = None
            hit_iou = 0.0

            candidate_event_i = assigned_event_i
            # while the events keep (potentially) intersecting, keep iterating and checking the intersection criterion
            while candidate_event_i < len(assigned_events) and \
                  assigned_events[candidate_event_i].start < ground_truth_event.end:
                intersection_flag, iou = check_event_intersection(ground_truth_event, assigned_events[candidate_event_i],
                                                                  intersection_over_union_threshold=
                                                                  intersection_over_union_threshold,
                                                                  return_iou=True)
                if intersection_flag:
                    # found the valid intersection of events, but is it the highest IoU?
                    if iou > hit_iou:
                        hit_event_i = candidate_event_i
                        hit_iou = iou
                else:
                    # intersection criteria are not fulfilled, but maybe there are better-intersected events ahead
                    pass
                candidate_event_i += 1

            if hit_event_i is None:
                # no match found -> Miss
                missed_labels_buffer.append(ground_truth_event.type)
            else:
                # Found some match, set all events between @assigned_event_i and @hit_event_i as false alarms
                for candidate_event_i in range(assigned_event_i, hit_event_i):
                    false_alarm_labels_buffer.append(assigned_events[candidate_event_i].type)
                # Set the @hit_i event as a hit with @ground_truth_event
                true_labels_events.append(ground_truth_event.type)
                assigned_labels_events.append(assigned_events[hit_event_i].type)
                local_matched_events_count += 1
                # Move @assigned_event_i to @hit_i + 1 and start further processing there
                assigned_event_i = hit_event_i + 1

        # went through all the ground truth events, let's see whether any detected events remain (all False Alarms)
        for candidate_event_i in range(assigned_event_i, len(assigned_events)):
            false_alarm_labels_buffer.append(assigned_events[candidate_event_i].type)

        assert len(ground_truth_events) + len(assigned_events) == local_matched_events_count * 2 + \
                                                                  len(missed_labels_buffer) + \
                                                                  len(false_alarm_labels_buffer), \
            'Different number of events matched + not matched compared to the total number of events: {} events ' \
            'in two lists in total, {} -- after matching'.format(len(ground_truth_events) + len(assigned_events),
                                                                 local_matched_events_count * 2 +
                                                                 len(missed_labels_buffer) +
                                                                 len(false_alarm_labels_buffer))
        # add missed events
        true_labels_events += missed_labels_buffer
        assigned_labels_events += ['UNKNOWN'] * len(missed_labels_buffer)
        # add false alarm events
        true_labels_events += ['UNKNOWN'] * len(false_alarm_labels_buffer)
        assigned_labels_events += false_alarm_labels_buffer

    assert len(true_labels_events) == len(assigned_labels_events)

    if positive_label is not None:
        true_labels_events = [x if x == positive_label else '_WRONG_LABEL' for x in true_labels_events]
        assigned_labels_events = [x if x == positive_label else '_WRONG_LABEL' for x in assigned_labels_events]

    stats = {'kappa': cohen_kappa_score(true_labels_events, assigned_labels_events)}
    return stats


def evaluate_episodes_as_Hooge_et_al(true_labels_list,
                                     assigned_labels_list,
                                     experts, positive_label='SP',
                                     intersection_over_union_threshold=0.0,
                                     return_raw_stats=False,
                                     verbose=False):
    """
    Event-level F1 scores [1] evaluation of the labelling result (algorithm output) in @assigned_labels_list with
    hand-labelling expert's labels in @ground_truth_list. In addition to [1], added an option to limit event hits
    to those with a good intersection-over-union score only (stricter evaluation)

    :param true_labels_list: list of arff objects produced with hand-labelling tool [2].
    :param assigned_labels_list: list of arff objects produced with this tool (or loaded via RecordingProcessor).
    :param experts: list of experts (for our data, one expert was the tie-corrector, so normally a list of one element
                    should be used, ['handlabeller_final']).
    :param intersection_over_union_threshold: (has to be a floating point number in the range of [0; 1]) only count
                                              a "hit", if the IoU is no smaller than this threshold
    :param positive_label: the positive abel to be evaluated (normally would be 'SP'/'FIX'/'SACCADE').
    :param return_raw_stats: whether to return raw statistics (TP/FP/FN stats) or the nicer F1 scores
    :param verbose: output runtime (debug) information
    :return: evaluation results in a dictionary form

    [1] https://link.springer.com/article/10.3758/s13428-017-0955-x
    [2] http://ieeexplore.ieee.org/abstract/document/7851169/
    """
    raw_stats = {
        'TP': 0.0,
        'FP': 0.0,
        'FN': 0.0,
        'Total IoU': 0.0,
        'Total events': 0.0,
        'Total detected events': 0.0
    }

    # recover the proper names of the events from hand-labelled data with the default scheme
    mapping_labels_to_names = EM_VALUE_MAPPING_DEFAULT

    for ground_truth, assigned_labels in zip(true_labels_list, assigned_labels_list):
        # check that the t-x-y data has all at least similar values
        assert np.allclose(ground_truth['data']['time'], assigned_labels['data']['time'])
        assert np.allclose(ground_truth['data']['x'], assigned_labels['data']['x'])
        assert np.allclose(ground_truth['data']['y'], assigned_labels['data']['y'])

        ground_truth_labels = get_majority_vote(ground_truth, experts)
        ground_truth_events = extract_events(ground_truth_labels, type_mapping_dict=mapping_labels_to_names)
        assigned_events = extract_events(assigned_labels['data']['EYE_MOVEMENT_TYPE'])

        # only keep the relevant events
        ground_truth_events = [x for x in ground_truth_events if x.type == positive_label]
        assigned_events = [x for x in assigned_events if x.type == positive_label]
        raw_stats['Total detected events'] += len(assigned_events)

        assigned_event_i = 0
        for ground_truth_event in ground_truth_events:
            raw_stats['Total events'] += 1
            # find the intersecting assigned events
            # skip through the events that end before the current ground truth one
            while assigned_event_i < len(assigned_events) and \
                  assigned_events[assigned_event_i].end <= ground_truth_event.start:
                assigned_event_i += 1
                raw_stats['FP'] += 1  # we had to skip a detected event because it didn't match anything -> False Alarm
                if verbose:
                    print('Registered a False Alarm for', assigned_events[assigned_event_i - 1], file=sys.stderr)

            hit_event_i = None
            hit_iou = 0.0
            # while the events keep (potentially) intersecting, keep iterating and checking the intersection criterion
            while assigned_event_i < len(assigned_events) and \
                  assigned_events[assigned_event_i].start < ground_truth_event.end:
                intersection_flag, iou = check_event_intersection(ground_truth_event, assigned_events[assigned_event_i],
                                                                  intersection_over_union_threshold=
                                                                  intersection_over_union_threshold,
                                                                  return_iou=True)
                if intersection_flag:
                    # found the intersection
                    hit_event_i = assigned_event_i
                    hit_iou = iou

                    # this event is taken now, moving on
                    assigned_event_i += 1
                    raw_stats['TP'] += 1  # found a match -> Hit
                    if verbose:
                        print('Registered a Hit for', ground_truth_event, 'and', assigned_events[assigned_event_i - 1], file=sys.stderr)

                    break
                else:
                    assigned_event_i += 1
                    raw_stats['FP'] += 1  # we had to skip a detected event because it didn't match anything -> False Alarm
                    if verbose:
                        print('Registered a False Alarm for', assigned_events[assigned_event_i - 1], file=sys.stderr)

            if hit_event_i is None:
                raw_stats['FN'] += 1  # no match found -> Miss
                if verbose:
                    print('Registered a Miss for', ground_truth_event, file=sys.stderr)
            raw_stats['Total IoU'] += hit_iou  # 0 if no match was found

        # went through all the ground truth events, let's see whether any detected events remain (all False Alarms)
        if assigned_event_i < len(assigned_events):
            raw_stats['FP'] += len(assigned_events) - assigned_event_i
            if verbose:
                print('Registered', len(assigned_events) - assigned_event_i, 'additional False Alarms', file=sys.stderr)

    if return_raw_stats:
        return raw_stats
    else:
        stats = {
            'F1': (2 * raw_stats['TP'] / (2 * raw_stats['TP'] + raw_stats['FN'] + raw_stats['FP']))
                  if 2 * raw_stats['TP'] + raw_stats['FN'] + raw_stats['FP'] != 0 else 0.0,
            'IoU': (raw_stats['Total IoU'] / raw_stats['Total events']) if raw_stats['Total events'] != 0 else 0.0
        }
        return stats


def evaluate_episodes_as_Hoppe_et_al(true_labels_list, assigned_labels_list, experts, positive_label='SP',
                                     return_raw_stats=False, interval_vs_interval=False):
    """
    Event-level evaluation of labelling result (algorithm output) in @assigned_labels_list with
    hand-labelling expert's labels in @ground_truth_list. Approximately following the event evaluation
    strategy of Hoppe and Bulling, 2016 [2], except for adding a possibility (via @interval_vs_interval) to
    enforce the matching of episodes to episodes, not just checking that the majority of samples are of some
    label type.

    :param true_labels_list: list of arff objects produced with hand-labelling tool [1].
    :param assigned_labels_list: list of arff objects produced with this tool (or loaded via RecordingProcessor).
    :param experts: list of experts (for our data, one expert was the tie-corrector, so normally a list of one element
                    should be used, ['handlabeller_final']).
    :param positive_label: the positive abel to be evaluated (normally would be 'SP'/'FIX'/'SACCADE').
    :param return_raw_stats: whether to return raw statistics (a confusion matrix) or nicer (per-class accuracy) ones
    :param interval_vs_interval: if False, will check if EM episodes in the ground truth are covered by at least 50%
                                 of correct labels;
                                 if True, will check if at least 50% of those episodes is covered by a continuous
                                 interval of the correct label.
    :return: evaluation results in a dictionary form

    [1] http://ieeexplore.ieee.org/abstract/document/7851169/
    [2] https://arxiv.org/abs/1609.02452
    """
    # get all possible labels for the confusion matrix
    if len(true_labels_list) > 0 and true_labels_list[0]['data'][experts[0]].dtype.type is not np.string_:
        # dealing with non-categorical labels, use standard order of labels
        labels = ['FIX', 'SACCADE', 'SP', 'NOISE']
    else:
        labels = [set(obj['data']['EYE_MOVEMENT_TYPE']) for obj in assigned_labels_list]
        labels = list(set().union(*labels))
    raw_confusion = {k: 0.0 for k in labels}  # count the number of hits for each class
    raw_confusion_denominator = 0.0

    raw_stats = {
        'TP': 0.0,
        'FP': 0.0,
        'TN': 0.0,
        'FN': 0.0
    }

    for ground_truth, assigned_labels in zip(true_labels_list, assigned_labels_list):
        # check that the t-x-y data has all at least similar values
        assert np.allclose(ground_truth['data']['time'], assigned_labels['data']['time'])
        assert np.allclose(ground_truth['data']['x'], assigned_labels['data']['x'])
        assert np.allclose(ground_truth['data']['y'], assigned_labels['data']['y'])

        ground_truth_labels = get_majority_vote(ground_truth, experts)

        start_i = 0
        for current_label, grp in itertools.groupby(ground_truth_labels):
            grp_len = len(list(grp))

            alg_labels = assigned_labels['data']['EYE_MOVEMENT_TYPE'][start_i:start_i + grp_len]
            for equivalent_noise_label in ['NOISE_CLUSTER', 'BLINK']:
                alg_labels[alg_labels == equivalent_noise_label] = 'NOISE'

            if not interval_vs_interval:  # check if at least 50% of the interval is covered by correct labels
                alg_labels = Counter(alg_labels)
                alg_majority_label, alg_majority_size = alg_labels.most_common(1)[0]
                if alg_majority_size >= 0.5 * grp_len:
                    # a hit for the current label in the @alg_majority_label column
                    pass
                else:
                    alg_majority_label = 'UNKNOWN'  # no label being assigned
            else:  # check if at least 50% of the interval is covered by a continuous interval of correct labels
                label_groups = itertools.groupby(alg_labels)
                label_groups = [(k, len(list(v))) for k, v in label_groups]
                label_values = [k for k, _ in label_groups]
                label_counts = [v for _, v in label_groups]

                max_i = np.argmax(label_counts)
                if label_counts[max_i] >= 0.5 * grp_len:
                    alg_majority_label = label_values[max_i]
                else:
                    alg_majority_label = 'UNKNOWN'

            # Record confusion matrix row and the @raw_stats.
            # Ensure that the @current_label - 1 is in tha valid range
            # (otherwise, it is some extra label, like PSO, which we ignore).
            if (isinstance(current_label, str) and current_label == positive_label) or \
                    (np.issubdtype(type(current_label), np.integer) and (0 <= current_label - 1 < len(labels))
                     and labels[current_label - 1] == positive_label):
                raw_confusion_denominator += 1
                if alg_majority_label in labels:
                    raw_confusion[alg_majority_label] += 1
                elif alg_majority_label != 'UNKNOWN':
                    print('Had to skip this label when computing the confusion matrix: ' \
                                         '{}, while full label list contains {} (this should not happen!)'.\
                        format(alg_majority_label, labels), file=sys.stderr)

                if alg_majority_label == positive_label:
                    # true: +, detected: +
                    raw_stats['TP'] += 1
                else:
                    # true: +, detected: -
                    raw_stats['FN'] += 1
            else:
                if alg_majority_label == positive_label:
                    # true: -, detected: +
                    raw_stats['FP'] += 1
                else:
                    # true: -, detected: -
                    raw_stats['TN'] += 1

            start_i += grp_len

    raw_confusion = {k: (raw_confusion[k] / raw_confusion_denominator) if raw_confusion_denominator != 0 else 0.0 for k in labels}

    if return_raw_stats:
        raw_stats['confusion'] = raw_confusion
        return raw_stats
    else:
        stats = compute_statistics(raw_stats)
        stats['confusion'] = raw_confusion
        stats['confusion-accuracy'] = raw_confusion.get(positive_label, 0.0)
        return stats


def evaluate_samples(true_labels_list, assigned_labels_list, experts, positive_label='SP', return_raw_stats=False):
    """
    Evaluate labelling result (algorithm output) in @assigned_labels_list with hand-labelling expert's labels
    in @ground_truth_list.

    :param true_labels_list: list of arff objects produced with hand-labelling tool [1].
    :param assigned_labels_list: list of arff objects produced with this tool (or loaded via RecordingProcessor).
    :param experts: list of experts (for our data, one expert was the tie-corrector, so normally a list of one element
                    should be used).
    :param positive_label: the positive abel to be evaluated (normally would be 'SP'/'FIX'/'SACCADE').
    :param return_raw_stats: whether to return raw statistics (TP/TN/FP/FN) or nicer (F1/precision/recall/...) ones
    :return: evaluation results in a dictionary form

    [1] http://ieeexplore.ieee.org/abstract/document/7851169/
    """
    raw_stats = {
        'TP': 0.,
        'FP': 0.,
        'TN': 0.,
        'FN': 0.
    }
    for ground_truth, assigned_labels in zip(true_labels_list, assigned_labels_list):
        # check that the t-x-y data has all at least similar values
        assert np.allclose(ground_truth['data']['time'], assigned_labels['data']['time'])
        assert np.allclose(ground_truth['data']['x'], assigned_labels['data']['x'])
        assert np.allclose(ground_truth['data']['y'], assigned_labels['data']['y'])

        if positive_label is not None:
            assigned_labels_status_list = (assigned_labels['data']['EYE_MOVEMENT_TYPE'] == positive_label).astype(int)

            ground_truth_status_list = get_majority_vote_efficient(ground_truth, experts, positive_label)
            # ground_truth_status_list = (get_majority_vote(ground_truth, experts) ==
            #                             CORRESPONDENCE_TO_HAND_LABELLING_VALUES[positive_label]).astype(int)

            raw_stats['TP'] += ((ground_truth_status_list == 1) * (assigned_labels_status_list == 1)).sum()
            raw_stats['FP'] += ((ground_truth_status_list == 0) * (assigned_labels_status_list == 1)).sum()
            raw_stats['TN'] += ((ground_truth_status_list == 0) * (assigned_labels_status_list == 0)).sum()
            raw_stats['FN'] += ((ground_truth_status_list == 1) * (assigned_labels_status_list == 0)).sum()
        else:
            ground_truth_labels = np.array(get_majority_vote(ground_truth, experts))
            if not ground_truth_labels.dtype.name.startswith('str'):
                ground_truth_labels = [EM_VALUE_MAPPING_DEFAULT[x] for x in ground_truth_labels]
            ground_truth_labels = np.array(ground_truth_labels)
            raw_stats['TP'] += (ground_truth_labels == assigned_labels['data']['EYE_MOVEMENT_TYPE']).sum()
            raw_stats['FP'] += (ground_truth_labels != assigned_labels['data']['EYE_MOVEMENT_TYPE']).sum()

    if return_raw_stats:
        stats = raw_stats
    else:
        stats = compute_statistics(raw_stats)
    return stats


def evaluate(true_labels_list, assigned_labels_list, experts, positive_label='SP', return_raw_stats=False,
             microseconds_in_time_unit=1.0, verbose=False):
    """
    Evaluate labelling result (algorithm output) in @assigned_labels_list with hand-labelling expert's labels
    in @ground_truth_list.

    :param true_labels_list: list of arff objects produced with hand-labelling tool [1].
    :param assigned_labels_list: list of arff objects produced with this tool (or loaded via RecordingProcessor).
    :param experts: list of experts (for our data, one expert was the tie-corrector, so normally a list of one element
                    should be used).
    :param positive_label: the positive abel to be evaluated (normally would be 'SP'/'FIX'/'SACCADE').
    :param return_raw_stats: whether to return raw statistics (TP/TN/FP/FN) or nicer (F1/precision/recall/...) ones
    :param microseconds_in_time_unit: how many microseconds in one time unit (the "time" column of the .arff files);
                                       needed to compute the correct event durations in ms; 1.0 for GazeCom
    :param verbose: output runtime (debug) information
    :return: evaluation results in a dictionary form

    [1] http://ieeexplore.ieee.org/abstract/document/7851169/
    """
    stats = {}

    stats.update(evaluate_samples(true_labels_list=true_labels_list,
                                  assigned_labels_list=assigned_labels_list,
                                  experts=experts,
                                  positive_label=positive_label,
                                  return_raw_stats=return_raw_stats))
    # these do not run with a None for a @positive_label
    if positive_label is not None:
        stats['episode_as_Hoppe_et_al'] = evaluate_episodes_as_Hoppe_et_al(true_labels_list=true_labels_list,
                                                                           assigned_labels_list=assigned_labels_list,
                                                                           experts=experts,
                                                                           positive_label=positive_label,
                                                                           return_raw_stats=return_raw_stats,
                                                                           interval_vs_interval=False)

        # Outdated evaluation, can be enabled by simply uncommenting, if necessary
        # stats['episode-vs-episode'] = evaluate_episodes_as_Hoppe_et_al(true_labels_list=true_labels_list,
        #                                                 assigned_labels_list=assigned_labels_list,
        #                                                 experts=experts,
        #                                                 positive_label=positive_label,
        #                                                 return_raw_stats=return_raw_stats,
        #                                                 interval_vs_interval=True)

        stats['episode_as_Hooge_et_al'] = evaluate_episodes_as_Hooge_et_al(true_labels_list=true_labels_list,
                                                                           assigned_labels_list=assigned_labels_list,
                                                                           experts=experts,
                                                                           positive_label=positive_label,
                                                                           return_raw_stats=return_raw_stats,
                                                                           intersection_over_union_threshold=0.0,
                                                                           verbose=verbose)

        if False:
            for iou_thd in np.arange(0, 1.05, 0.05):
                stats['episode_as_Hooge_et_al']['IoU>={}'.format(iou_thd)] = evaluate_episodes_as_Hooge_et_al(
                    true_labels_list=true_labels_list,
                    assigned_labels_list=assigned_labels_list,
                    experts=experts,
                    positive_label=positive_label,
                    return_raw_stats=return_raw_stats,
                    intersection_over_union_threshold=iou_thd,
                    verbose=verbose)

        else:
            stats['episode_as_Hooge_et_al']['IoU>=0.5'] = evaluate_episodes_as_Hooge_et_al(
                true_labels_list=true_labels_list,
                assigned_labels_list=assigned_labels_list,
                experts=experts,
                positive_label=positive_label,
                return_raw_stats=return_raw_stats,
                intersection_over_union_threshold=0.5,
                verbose=verbose)

    # these work fine with a None @positive_label
    stats['basic_statistics'] = evaluate_basic_statistics(true_labels_list=true_labels_list,
                                                          assigned_labels_list=assigned_labels_list,
                                                          experts=experts,
                                                          positive_label=positive_label,
                                                          return_raw_stats=return_raw_stats,
                                                          microseconds_in_time_unit=microseconds_in_time_unit)

    stats['episode_as_Zemblys_et_al'] = evaluate_episodes_as_Zemblys_et_al(true_labels_list=true_labels_list,
                                                                           assigned_labels_list=assigned_labels_list,
                                                                           experts=experts,
                                                                           positive_label=positive_label,
                                                                           intersection_over_union_threshold=0.0,
                                                                           verbose=verbose)
    stats['episode_as_Zemblys_et_al']['IoU>=0.5'] = evaluate_episodes_as_Zemblys_et_al(
        true_labels_list=true_labels_list,
        assigned_labels_list=assigned_labels_list,
        experts=experts,
        positive_label=positive_label,
        intersection_over_union_threshold=0.5,
        verbose=verbose)

    stats['episode_adjusted_Cohens_kappa'] = evaluate_episodes_adjusted_Cohens_kappa(
        true_labels_list=true_labels_list,
        assigned_labels_list=assigned_labels_list,
        experts=experts,
        positive_label=positive_label,
        intersection_over_union_threshold=0.0,
        return_raw_stats=return_raw_stats,
        verbose=verbose)

    stats['episode_adjusted_Cohens_kappa']['IoU>=0.8'] = evaluate_episodes_adjusted_Cohens_kappa(
        true_labels_list=true_labels_list,
        assigned_labels_list=assigned_labels_list,
        experts=experts,
        positive_label=positive_label,
        intersection_over_union_threshold=0.8,
        return_raw_stats=return_raw_stats,
        verbose=verbose)

    stats['normalised_Levenshtein'] = evaluate_normalised_Levenshtein_dist(true_labels_list=true_labels_list,
                                                                           assigned_labels_list=assigned_labels_list,
                                                                           experts=experts,
                                                                           positive_label=positive_label,
                                                                           return_raw_stats=return_raw_stats)
    return stats
