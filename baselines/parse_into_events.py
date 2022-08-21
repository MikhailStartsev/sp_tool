#!/usr/bin/env python
import glob
from argparse import ArgumentParser
import sys, os
import numpy as np
from itertools import groupby

from sp_tool import recording_processor
from supersaliency import util


def generate_permutation(args):
    all_folders = sorted(glob.glob('{}/*/'.format(args.folder)))
    all_folders = [os.path.split(x.rstrip('/'))[1] for x in all_folders]

    permutation = all_folders

    if args.permutation:
        np.random.seed(0)
        while any([x == y for x, y in util.zip_equal(all_folders, permutation)]):
            permutation = np.random.permutation(all_folders)

    return dict(util.zip_equal(all_folders, permutation))


def dump_events_as_csv(args):
    rp = recording_processor.RecordingProcessor()
    all_filenames = sorted(glob.glob('{}/*/*.arff'.format(args.folder)))
    label_objects = rp.load_multiple_recordings(all_filenames,
                                                labelled_eye_movement_column_arff=args.column)
    out = sys.stdout
    if args.csv:
        out = open(args.csv, 'w')
    header = ['em_type', 'duration_ms', 'amplitude_deg',
              'preceding_em', 'successive_em',
              'onset_x_deg', 'onset_y_deg', 'onset_t_ms', 'offset_x_deg', 'offset_y_deg', 'offset_t_ms',
              'clip_name', 'observer_name']
    if args.samples_duration:
        header.append('duration_samples')
    print(','.join(header), file=out)

    whole_data = []
    ppd = util.datasets.get_ppd('GazeCom')

    folders_permutation = generate_permutation(args)
    print('Using the following folder permutation:', folders_permutation, file=sys.stderr)

    for filename, obj in util.zip_equal(all_filenames, label_objects):
        obj = obj['data']
        dtype = obj.dtype

        folder, filename = os.path.split(filename)
        filename = os.path.splitext(filename)[0]

        observer, _ = filename.split('_', 1)

        clip = os.path.split(folder)[1]
        clip = folders_permutation[clip]

        grouping = [(em_type, np.array(list(group), dtype=dtype)) for em_type, group in
                    groupby(obj, key=lambda x: x['EYE_MOVEMENT_TYPE'])]
        for i, (em_type, group_matrix) in enumerate(grouping):
            if args.keep_noise or em_type != 'NOISE':
                duration = (group_matrix['time'][-1] - group_matrix['time'][0]) * 1e-3
                amplitude = np.linalg.norm([group_matrix['x'][0] - group_matrix['x'][-1],
                                            group_matrix['y'][0] - group_matrix['y'][-1]]) / ppd
                one_csv_line = {'em_type': em_type,
                                'duration_ms': duration,
                                'amplitude_deg': amplitude,
                                'onset_x_deg': group_matrix['x'][0] / ppd,
                                'onset_y_deg': group_matrix['y'][0] / ppd,
                                'onset_t_ms': group_matrix['time'][0] * 1e-3,
                                'offset_x_deg': group_matrix['x'][-1] / ppd,
                                'offset_y_deg': group_matrix['y'][-1] / ppd,
                                'offset_t_ms': group_matrix['time'][-1] * 1e-3,
                                'clip_name': clip,
                                'observer_name': observer,
                                'preceding_em': grouping[i - 1][0] if i > 0 else 'None',
                                'successive_em': grouping[i + 1][0] if i < len(grouping) - 1 else 'None'
                                }
                if args.samples_duration:
                    one_csv_line['duration_samples'] = len(group_matrix)
                print(','.join(map(str, [one_csv_line[x] for x in header])), file=out)
                whole_data.append(one_csv_line)

    if args.csv:
        out.close()
    else:
        out.flush()
    return whole_data


def parse_args():
    parser = ArgumentParser('Event parser')
    parser.add_argument('--in', '--folder', '-i', '-f', dest='folder', required=True,
                        help='Algorithm outputs folder, used as input here.')
    parser.add_argument('--column', '--col', '-c', default='EYE_MOVEMENT_TYPE',
                        help='ARFF file column with the labels to be analysed.')
    parser.add_argument('--csv',
                        help='The output .csv file. If none provided, will output to STDOUT.')
    parser.add_argument('--permutation', '--baseline', action='store_true',
                        help='This flag will toggle the `permutation baseline` mode, where we exchange randomly '
                             'exchange video names so that no video has "its own" events.')
    parser.add_argument('--keep-noise', action='store_true',
                        help='Keep the NOISE events in the data')
    parser.add_argument('--samples-duration', action='store_true',
                        help='Include duration in samples')
    return parser.parse_args()


def __main__():
    args = parse_args()
    dump_events_as_csv(args)


if __name__ == '__main__':
    __main__()
