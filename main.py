import argparse
from binary_optimization import *

                
""" 
TODO:
If we make this problem being able to buy multiple of each item (constrained by max)
it's kind of like multiobjective multidimensional optimization problem except with 
linear constraints. 
Could try simplex algorithm then. Or other algorithms. 
"""

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('problem')
    parser.add_argument('method')
    args = parser.parse_args()
    if args.problem == 'binary':
        if args.method == 'quad': 
            quadratic_penalty_constraint_binary()
        elif args.method == 'hooke': 
            hooke_jeeves_binary() 


if __name__ == '__main__': 
    main() 