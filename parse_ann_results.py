#!/bin/python3

import argparse
import pandas as pd
import re
import subprocess
import subprocess

def parse_args():
    parser = argparse.ArgumentParser("Format it nicely")

    parser.add_argument(
        '-f', '--files',
        type=str,
        required=True,
        help='Files with log from ann.py',
        dest='files',
        nargs='+'
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
    data_inners = []
    data_outers = []
    
    for f in args.files:
        # duplicate input file
        duplicate_file = f + '.COPY'
        subprocess.run(['cp', f, duplicate_file])

        with open(duplicate_file, 'r') as f:
            content = f.readlines()

        p_fold = re.compile(r'.+\s(\d)\/\d')
        p_loss = re.compile(r'.+:\s(\d+\.\d+).+:\s(\d+\.\d+)')
        p_best_inner = re.compile(r'.+loss\:\s(\d+\.\d+).+\s(\{.+\})')

        for line in content:
            line = line.strip()

            if line.startswith('Outer CV fold'):
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
                
    inner_df = pd.DataFrame(data_inners)
    inner_df.to_csv(args.prefix + 'ann_results_inner.csv', index=None)

    # by each outer_inner combi, find lowest eval_loss 
    # as this is the best inner_loss 
    print("Inner results:")
    inner_bests_idx = inner_df.groupby(['outer_fold', 'inner_fold'])['inner_eval_loss'].idxmin()
    print(inner_df.iloc[inner_bests_idx])

    # find best outer
    parameter_cols = ['out_units', 'lr', 'optimizer', 'p_dropout', 'use_early_stopping', 'momentum', 'l2_reg']
    outer_res = pd.DataFrame(data_outers)
    outer_mins = inner_df.groupby(['outer_fold'])['inner_eval_loss'].idxmin().values
    outer_df = inner_df.iloc[outer_mins].copy().merge(
        outer_res, 
        on=parameter_cols + [
            'epochs', 'activation_function', 'batch_size', 'outer_fold'
        ],
        how='left'
    )
    outer_df.drop(columns=[x for x in outer_df.columns if x.endswith('_y')], inplace=True)
    outer_df = outer_df.groupby(['outer_fold'] + parameter_cols)[['inner_train_loss', 'inner_eval_loss_x', 'outer_train_loss', 'outer_eval_loss']].mean().reset_index()
    outer_df.to_csv(args.prefix + 'ann_results_outer.csv', index=None)
    print("Outer results")
    print(outer_df)
    
    sh_script_name = args.prefix + 'run_fold_anns.sh'
    sh_script = open(sh_script_name, 'w')
    print("#!/bin/bash", file=sh_script)
    py_str = "python3 ann_hpc.py -out_units {o} -lr {lr} -p_dropout {p} -optimizer {opt} -momentum {m} -l2_reg {l2} -es {es} -log {log} -cp {cp}"
    for _, row in outer_df.iterrows():
        outstr = py_str.format(
            o=row['out_units'],
            lr=row['lr'],
            p=row['p_dropout'],
            opt=row['optimizer'],
            m=row['momentum'],
            l2=row['l2_reg'],
            es=int(row['use_early_stopping']),
            log=args.prefix + 'ann_fold{}.txt'.format(row['outer_fold']),
            cp=args.prefix + 'ann_fold{}.pt'.format(row['outer_fold'])
        )
        print(outstr, file=sh_script)
    sh_script.close()

    subprocess.run(['chmod', '+x', sh_script_name])
