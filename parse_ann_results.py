#!/bin/python3

import argparse
import pandas as pd
import re
import subprocess

def parse_args():
    parser = argparse.ArgumentParser("Format it nicely")

    parser.add_argument(
        '-f', '--file',
        type=str,
        required=True,
        help='File with log from ann.py',
        dest='file'
    )
    parser.add_argument(
        '-p', '--prefix',
        type=str,
        required=True,
        help='Add prefix to output files written',
        dest='prefix'
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # duplicate input file
    duplicate_file = args.file + '.COPY'
    subprocess.run(['cp', args.file, duplicate_file])

    with open(duplicate_file, 'r') as f:
        content = f.readlines()

    data_inners = []
    data_outers = []

    p_fold = re.compile(r'.+\s(\d)\/\d')
    p_loss = re.compile(r'.+:\s(\d+\.\d+).+:\s(\d+\.\d+)')
    p_best_inner = re.compile(r'.+\:\s(\d+\.d+)\sparams\s(.+)$')

    for line in content:
        line = line.strip()

        if line.startswith('Outer'):
            m = p_fold.match(line)
            outer_fold = int(m.group(1))

        elif line.startswith('> Inner'):
            m = p_fold.match(line)
            inner_fold = int(m.group(1))
        elif line.startswith('>> params:'):
            params = eval(line.replace('>> params: ', ''))
            params['outer_fold'] = outer_fold
            params['inner_fold'] = inner_fold
        elif line.startswith('>>>'):
            m = p_loss.match(line)
            params['inner_train_loss'] = float(m.group(1))
            params['inner_eval_loss'] = float(m.group(2))

            data_inners.append(params)
            params = {}
        
        elif line.startswith('Best inner'):
            m = p_best_inner.match(line)
            outer_params = eval(m.group(2))
            outer_params['inner_eval_loss'] = float(m.group(1))
            outer_params['outer_fold'] = outer_fold
        
        elif line.startswith('Score on outer'):
            m = p_loss.match(line)
            outer_params['outer_train_loss'] = float(m.group(1))
            outer_params['outer_eval_loss'] = float(m.group(2))

            data_outers.append(outer_params)
            outer_params = {}
            
    df = pd.DataFrame(data_inners)
    df.to_csv(args.prefix + 'ann_results_inner.csv', index=None)

    # by each outer_inner combi, find lowest eval_loss 
    # as this is the best inner_loss 
    print("Inner results:")
    inner_bests_idx = df.groupby(['outer_fold', 'inner_fold'])['inner_eval_loss'].idxmin()
    print(df.iloc[inner_bests_idx])

    # outer
    print("Outer results")
    df = pd.DataFrame(data_outers)
    df.to_csv(args.prefix + 'ann_results_outer.csv', index=None)
    print(df)
    