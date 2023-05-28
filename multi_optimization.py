""" 
If we make this problem being able to buy multiple of each item (constrained by max)
it's multiobjective multidimensional optimization problem except with 
linear constraints. 
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


'''
Hooke Jeeves for multi optimization. 
We manually constrain the problem to prevent Hooke Jeeves from moving outside the constraint boundaries. 
'''
def hooke_jeeves_multi(file='shopping_list.txt', num_its=50, overbudget_penalty=0.05, spending_penalty=0.05): 
    student = CollegeStudent(file)
    shopping_list_len = len(student.shopping_list)
    best_pt = sample_point_multi(student)
    total_util, total_cost = calc_util_and_cost(best_pt, student.shopping_utils, student.shopping_costs)
    util_plus_penalty = calc_util_plus_penalty(total_util, total_cost, overbudget_penalty, spending_penalty, student.budget)
    for _ in range(num_its): 
        best_pt_after_it = best_pt 
        best_util, best_cost = total_util, total_cost
        best_util_plus_penalty = util_plus_penalty
        for dim in range(shopping_list_len): 
            for sign in [-1, 1]: 
                new_pt = best_pt.copy()
                new_value = best_pt[dim] + sign 
                if new_value < 0 or new_value > student.max_amounts[dim, :]: continue # manually constrain the problem 
                new_pt[dim] = new_value
                util, cost = calc_util_and_cost(new_pt, student.shopping_utils, student.shopping_costs)
                new_util_plus_penalty = calc_util_plus_penalty(util, cost, overbudget_penalty, spending_penalty, student.budget)
                if new_util_plus_penalty > util_plus_penalty: 
                    best_util_plus_penalty = new_util_plus_penalty
                    best_pt_after_it = new_pt
                    best_util, best_cost = util, cost 
        # only update everything after exploring all dimensions in an iteration 
        total_util, total_cost = best_util, best_cost
        util_plus_penalty = best_util_plus_penalty
        best_pt = best_pt_after_it
    print_results(student, best_pt, total_cost, total_util)

def transition_to_new_pt(old_pt, student: CollegeStudent, threshold=0.3): 
    new_pt = np.zeros(old_pt.shape)
    for i in range(len(student.shopping_list)):
        if random.random() < threshold: 
            new_pt[i] = random.choice(range(int(student.max_amounts[i])+1)) # satisfy the constraints 
        else:
            new_pt[i] = old_pt[i]
    return new_pt 

def simulated_annealing_multi(file='shopping_list.txt', num_its=100, overbudget_penalty=500, spending_penalty=5, temp=5, decay=1.01):
    student = CollegeStudent(file) 
    best_pt = sample_point_multi(student)
    total_util, total_cost = calc_util_and_cost(best_pt, student.shopping_utils, student.shopping_costs)
    best_util_plus_penalty = calc_util_plus_penalty(total_util, total_cost, overbudget_penalty, spending_penalty, student.budget)
    cur_pt = best_pt
    cur_util_plus_penalty = best_util_plus_penalty
    for _ in range(num_its): 
        new_pt = transition_to_new_pt(cur_pt, student)
        new_util, new_cost = calc_util_and_cost(new_pt, student.shopping_utils, student.shopping_costs)
        new_util_plus_penalty = calc_util_plus_penalty(new_util, new_cost, overbudget_penalty, spending_penalty, student.budget)
        if new_util_plus_penalty > cur_util_plus_penalty or random.random() < np.exp(- temp * decay):
            cur_pt = new_pt 
            cur_util_plus_penalty = new_util_plus_penalty
        if new_util_plus_penalty > best_util_plus_penalty:  # global update 
            best_util_plus_penalty = new_util_plus_penalty
            best_pt = new_pt 
            total_util, total_cost = new_util, new_cost 
    print_results(student, best_pt, total_cost, total_util)
