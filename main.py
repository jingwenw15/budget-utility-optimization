import argparse
from binary_optimization import *
from multi_optimization import * 

                

def main(): 
    random.seed(17)
    np.random.seed(17)
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('problem')
    parser.add_argument('method')
    parser.add_argument('plot')
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
            quad_penalty_multi(args.file, plot=args.plot)
        elif args.method == 'hooke':
            hooke_jeeves_multi(args.file, plot=args.plot)
        elif args.method == 'anneal':
            simulated_annealing_multi(args.file, plot=args.plot)


if __name__ == '__main__': 
    main() 