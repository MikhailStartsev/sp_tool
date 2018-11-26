#!/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np
import itertools
from collections import Counter, namedtuple
import sys
from sklearn.metrics import cohen_kappa_score

Event = namedtuple('Event', ['type', 'start', 'end', 'duration'])

# Ground truth is usually stored in ARFF files with a separate column for each hand-labelling expert (if multiple are
# present). Each hand-labelling expert column contains numerical data with the following correspondence to eye movement
# types:
CORRESPONDENCE_TO_HAND_LABELLING_VALUES = {
    'UNKNOWN': 0,
    'FIX': 1,
    'SACCADE': 2,
    'SP': 3,
    'NOISE': 4
}


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
    if len(experts) == 1:
        # just one expert, he always agrees with himself
        return obj['data'][experts[0]] == CORRESPONDENCE_TO_HAND_LABELLING_VALUES[positive_label]
    else:
        hand_labellings = obj['data'][experts]
        hand_labellings_list = hand_labellings.view((np.float32, len(hand_labellings.dtype.names)))
        thd = len(experts) / 2.0  # at least 50% of experts should agree
        majority_vote = ((hand_labellings_list == CORRESPONDENCE_TO_HAND_LABELLING_VALUES[positive_label]).sum(axis=1)
                         >= thd).astype(int)
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
    if len(experts) == 1:
        return obj['data'][experts].astype(int)
    else:
        if exclude_values is not None:
            if not isinstance(exclude_values, list):
                exclude_values = [exclude_values]
        else:
            exclude_values = []
        exclude_values = set(exclude_values)

        majority_vote = []
        hand_labellings = obj['data'][experts].tolist()
        for i in xrange(len(hand_labellings)):
            # max number of occurrences wins
            candidates_set = set(hand_labellings[i]).difference(exclude_values)
            if not candidates_set:
                candidates_set = set(hand_labellings[i])
            majority_vote.append(max(candidates_set, key=hand_labellings[i].count))
        majority_vote = np.asarray(majority_vote, dtype=np.int)

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
            event_type = type_mapping_dict[event_type]
        events.append(Event(type=event_type, start=current_i, end=current_i + event_len, duration=event_len))
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
    :param positive_label: the positive abel to be evaluated (normally would be 'SP'/'FIX'/'SACCADE').
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
    from data_loaders import load_ARFF_as_arff_object
    mapping_labels_to_names = load_ARFF_as_arff_object.EM_VALUE_MAPPING_DEFAULT

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
        true_labels_events = map(lambda x: x if x == positive_label else '_WRONG_LABEL', true_labels_events)
        assigned_labels_events = map(lambda x: x if x == positive_label else '_WRONG_LABEL', assigned_labels_events)

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
    from data_loaders import load_ARFF_as_arff_object
    mapping_labels_to_names = load_ARFF_as_arff_object.EM_VALUE_MAPPING_DEFAULT

    for ground_truth, assigned_labels in zip(true_labels_list, assigned_labels_list):
        # check that the t-x-y data has all at least similar values
        assert np.allclose(ground_truth['data']['time'], assigned_labels['data']['time'])
        assert np.allclose(ground_truth['data']['x'], assigned_labels['data']['x'])
        assert np.allclose(ground_truth['data']['y'], assigned_labels['data']['y'])

        ground_truth_labels = get_majority_vote(ground_truth, experts)
        ground_truth_events = extract_events(ground_truth_labels, type_mapping_dict=mapping_labels_to_names)
        assigned_events = extract_events(assigned_labels['data']['EYE_MOVEMENT_TYPE'])

        # only keep the relevant events
        ground_truth_events = filter(lambda x: x.type == positive_label, ground_truth_events)
        assigned_events = filter(lambda x: x.type == positive_label, assigned_events)
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
                    print >> sys.stderr, 'Registered a False Alarm for', assigned_events[assigned_event_i - 1]

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
                        print >> sys.stderr, 'Registered a Hit for', ground_truth_event, 'and', assigned_events[assigned_event_i - 1]

                    break
                else:
                    assigned_event_i += 1
                    raw_stats['FP'] += 1  # we had to skip a detected event because it didn't match anything -> False Alarm
                    if verbose:
                        print >> sys.stderr, 'Registered a False Alarm for', assigned_events[assigned_event_i - 1]

            if hit_event_i is None:
                raw_stats['FN'] += 1  # no match found -> Miss
                if verbose:
                    print >> sys.stderr, 'Registered a Miss for', ground_truth_event
            raw_stats['Total IoU'] += hit_iou  # 0 if no match was found

        # went through all the ground truth events, let's see whether any detected events remain (all False Alarms)
        if assigned_event_i < len(assigned_events):
            raw_stats['FP'] += len(assigned_events) - assigned_event_i
            if verbose:
                print >> sys.stderr, 'Registered', len(assigned_events) - assigned_event_i, 'additional False Alarms'

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
    labels = ['FIX', 'SACCADE', 'SP', 'NOISE']  # skip the UNKNOWN label
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
            if 0 <= current_label - 1 < len(labels) and labels[current_label - 1] == positive_label:
                raw_confusion_denominator += 1
                if alg_majority_label in labels:
                    raw_confusion[alg_majority_label] += 1

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
        stats['confusion-accuracy'] = raw_confusion[positive_label]
        return stats


def evaluate(true_labels_list, assigned_labels_list, experts, positive_label='SP', return_raw_stats=False,
             verbose=False):
    """
    Evaluate labelling result (algorithm output) in @assigned_labels_list with hand-labelling expert's labels
    in @ground_truth_list.

    :param true_labels_list: list of arff objects produced with hand-labelling tool [1].
    :param assigned_labels_list: list of arff objects produced with this tool (or loaded via RecordingProcessor).
    :param experts: list of experts (for our data, one expert was the tie-corrector, so normally a list of one element
                    should be used).
    :param positive_label: the positive abel to be evaluated (normally would be 'SP'/'FIX'/'SACCADE').
    :param return_raw_stats: whether to return raw statistics (TP/TN/FP/FN) or nicer (F1/precision/recall/...) ones
    :param verbose: output runtime (debug) information
    :return: evaluation results in a dictionary form

    [1] http://ieeexplore.ieee.org/abstract/document/7851169/
    """
    raw_stats = {
        'TP': 0,
        'FP': 0,
        'TN': 0,
        'FN': 0
    }
    for ground_truth, assigned_labels in zip(true_labels_list, assigned_labels_list):
        # check that the t-x-y data has all at least similar values
        assert np.allclose(ground_truth['data']['time'], assigned_labels['data']['time'])
        assert np.allclose(ground_truth['data']['x'], assigned_labels['data']['x'])
        assert np.allclose(ground_truth['data']['y'], assigned_labels['data']['y'])

        assigned_labels_status_list = (assigned_labels['data']['EYE_MOVEMENT_TYPE'] == positive_label).astype(int)

        ground_truth_status_list = get_majority_vote_efficient(ground_truth, experts, positive_label)
        # ground_truth_status_list = (get_majority_vote(ground_truth, experts) ==
        #                             CORRESPONDENCE_TO_HAND_LABELLING_VALUES[positive_label]).astype(int)

        raw_stats['TP'] += ((ground_truth_status_list == 1) * (assigned_labels_status_list == 1)).sum()
        raw_stats['FP'] += ((ground_truth_status_list == 0) * (assigned_labels_status_list == 1)).sum()
        raw_stats['TN'] += ((ground_truth_status_list == 0) * (assigned_labels_status_list == 0)).sum()
        raw_stats['FN'] += ((ground_truth_status_list == 1) * (assigned_labels_status_list == 0)).sum()

    if return_raw_stats:
        stats = raw_stats
    else:
        stats = compute_statistics(raw_stats)

    stats['episode_as_Hoppe_et_al'] = evaluate_episodes_as_Hoppe_et_al(true_labels_list=true_labels_list,
                                                                       assigned_labels_list=assigned_labels_list,
                                                                       experts=experts,
                                                                       positive_label=positive_label,
                                                                       return_raw_stats=return_raw_stats,
                                                                       interval_vs_interval=False)

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

    if False:
        for iou_thd in np.arange(0, 1.05, 0.05):
            stats['episode_as_Hooge_et_al']['IoU>={}'.format(iou_thd)] = evaluate_episodes_as_Hooge_et_al(
                true_labels_list=true_labels_list,
                assigned_labels_list=
                assigned_labels_list,
                experts=experts,
                positive_label=positive_label,
                return_raw_stats=return_raw_stats,
                intersection_over_union_threshold=iou_thd,
                verbose=verbose)

    else:
        stats['episode_as_Hooge_et_al']['IoU>=0.5'] = evaluate_episodes_as_Hooge_et_al(true_labels_list=true_labels_list,
                                                                                   assigned_labels_list=
                                                                                   assigned_labels_list,
                                                                                   experts=experts,
                                                                                   positive_label=positive_label,
                                                                                   return_raw_stats=return_raw_stats,
                                                                                   intersection_over_union_threshold=0.5,
                                                                                   verbose=verbose)

    return stats
