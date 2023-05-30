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

    should_plot = True if args.plot == "True" else False 

    if args.problem == 'binary':
        if args.method == 'sample': 
            quadratic_penalty_sampling_binary(args.file)
        elif args.method == 'hooke': 
            hooke_jeeves_binary(args.file) 
        elif args.method == 'brute':
            brute_force_binary(args.file)
    elif args.problem == 'multi': 
        if args.method == 'sample': 
            quad_penalty_sampling_multi(args.file, plot=should_plot)
        elif args.method == 'hooke':
            hooke_jeeves_multi(args.file, plot=should_plot)
        elif args.method == 'anneal':
            simulated_annealing_multi(args.file, plot=should_plot)
        elif args.method == 'genetic': 
            genetic_algorithm_multi(args.file, plot=should_plot)


if __name__ == '__main__': 
    main() 