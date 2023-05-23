""" 
TODO:
If we make this problem being able to buy multiple of each item (constrained by max)
it's kind of like multiobjective multidimensional optimization problem except with 
linear constraints. 
Could try simplex algorithm then (actually would have to use slack vars to make it equality). Or other algorithms. 
"""
import random 
import numpy as np 
from CollegeStudent import CollegeStudent
from utils import *

'''
For each item, we buy it with probability equal to threshold. 
Note that in this method, we can buy up to the item limit. 
'''
def sample_point_multi(student: CollegeStudent, threshold=0.4): 
    # does not use previous distribution at all 
    pt = np.zeros((len(student.shopping_list), 1))
    for i in range(len(student.shopping_list)): 
        pt[i] = random.choice(range(int(student.max_amounts[i])+1))
    return pt 

'''
Quadratic penalty function with the possibility of buying multiple of each item,
up to a maximum amount specified in the shopping list. 
'''
def quad_penalty_multi(file='shopping_list.txt', num_its=2000, overbudget_penalty=1000, spending_penalty=500): 
    student = CollegeStudent(file)
    max_util_with_penalty = float('-inf')
    max_util = None 
    pt_cost = None 
    best_pt = None 
    for _ in range(num_its): 
        pt = sample_point_multi(student)
        total_util, total_cost = calc_util_and_cost(pt, student.shopping_utils, student.shopping_costs)
        util_plus_penalty = calc_util_plus_penalty(total_util, total_cost, overbudget_penalty, spending_penalty, student.budget)
        if util_plus_penalty > max_util_with_penalty: 
            max_util_with_penalty = util_plus_penalty
            max_util = total_util 
            best_pt = pt 
            pt_cost = total_cost 
    print_results(student, best_pt, pt_cost, max_util)
    return best_pt 