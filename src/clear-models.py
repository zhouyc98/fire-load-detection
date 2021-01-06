#!/usr/bin/python3.7

import argparse
import glob

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', type=str, default='0', help='cuda visible device id')
    parser.add_argument('-t', '--ap_thr', type=float, default=23, help='ap threshold to clear')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    c='' if args.cuda=='0' else args.cuda
    fns=glob.glob(f'./output{c}/*-ap*.pth')
    n_clean=0
    for fn in fns:
        ap=float(fn.split('-ap')[1][:4])
        if ap<args.ap_thr:
            with open(fn, 'r+') as fp:
                fp.truncate()
            n_clean+=1
    
    print(f'Clear {n_clean} models')
