#!/usr/bin/env python
from argparse import ArgumentParser

# (make sure to first run `python setup.py install --user` in the source directory)
from sp_tool import run_detection


def detect_eye_movements(out_folder=None, verbose=False):
    # Initializing with all default parameters from the config file
    # (can actually just skip config_file=... completely, the parameters are still default).
    # If needed, can pass extra parameters here, or modify the config file.
    params = run_detection.create_parameters(config_file='default_parameters.conf.json',
                                             output_folder=out_folder, verbose=verbose)
    run_detection.run_detection(params)


def parse_args():
    parser = ArgumentParser('SP detection tool')
    parser.add_argument('--output-folder', '--out', required=False, default=None,
                        help='Where to output the resulting labelled data '
                             '(if empty, will create a new temporary directory)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Whether to output some information '
                                                                     'about the progress of the run to STDERR')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    detect_eye_movements(out_folder=args.output_folder, verbose=args.verbose)
