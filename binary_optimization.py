from utils import calc_util_and_cost, calc_util_plus_penalty, print_results
from CollegeStudent import CollegeStudent
import random 
import numpy as np 


'''
Overbudget penalty: the penalty for spending money above the budget 
Spending penalty: the penalty for spending money in general 
We optimize for utility subject to the constraint of budget. 
Another "constraint" is the spending penalty (since our second objective is saving money). 
'''
def quadratic_penalty_constraint_binary(num_its=20, overbudget_penalty=1, spending_penalty=0.5): 
    student = CollegeStudent()
    max_util_with_penalty = float('-inf')
    max_util = None 
    pt_cost = None 
    best_pt = None 
    for _ in range(num_its): 
        pt = sample_point_binary(len(student.shopping_list))
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
For each item, we buy it with probability equal to threshold. 
Note that in this method, we can only buy 0 or 1 of each item. 
'''
def sample_point_binary(num_items, threshold=0.4): 
    # does not use previous distribution at all 
    pt = np.zeros((num_items, 1))
    for i in range(num_items): 
        keep_prob = random.random()
        pt[i, :] = 1 if keep_prob < threshold else 0 
    return pt 

       

def hooke_jeeves_binary(num_its=50, overbudget_penalty=0.05, spending_penalty=0.05): 
    student = CollegeStudent()
    shopping_list_len = len(student.shopping_list)
    pt = sample_point_binary(shopping_list_len)
    best_pt = pt 
    total_util, total_cost = calc_util_and_cost(pt, student.shopping_utils, student.shopping_costs)
    util_plus_penalty = calc_util_plus_penalty(total_util, total_cost, overbudget_penalty, spending_penalty, student.budget)
    for _ in range(num_its): 
        best_pt_after_it = best_pt 
        best_util, best_cost = total_util, total_cost
        best_util_plus_penalty = util_plus_penalty
        for dim in range(shopping_list_len): 
            new_pt = best_pt.copy()
            new_pt[dim] = 1 - best_pt[dim] # flip the purchase
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
    