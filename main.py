import argparse
from binary_optimization import *
from multi_optimization import * 

                

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('problem')
    parser.add_argument('method')
    args = parser.parse_args()
    if args.problem == 'binary':
        if args.method == 'quad': 
            quadratic_penalty_constraint_binary(args.file)
        elif args.method == 'hooke': 
            hooke_jeeves_binary(args.file) 
        elif args.method == 'brute':
            brute_force_binary(args.file)
    elif args.problem == 'multi': 
        if args.method == 'quad': 
            quad_penalty_multi()


if __name__ == '__main__': 
    main() 