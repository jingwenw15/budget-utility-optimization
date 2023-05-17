from CollegeStudent import CollegeStudent
import numpy as np 
import random 

'''
Overbudget penalty: the penalty for spending money above the budget 
Spending penalty: the penalty for spending money in general 
'''
def quadratic_penalty_constraint_method(num_its=20, overbudget_penalty=500, spending_penalty=0.5): 
    student = CollegeStudent()
    max_util_with_penalty = float('-inf')
    max_util = None 
    pt_cost = None 
    best_pt = None 
    for _ in range(num_its): 
        pt = sample_point(len(student.shopping_list))
        total_util = np.sum(pt * student.shopping_utils) 
        total_cost = np.sum(pt * student.shopping_costs)
        util_plus_penalty = total_util - overbudget_penalty * (max(total_cost - student.budget, 0) ** 2) - ((spending_penalty * total_cost) ** 2)
        if util_plus_penalty > max_util_with_penalty: 
            max_util_with_penalty = util_plus_penalty
            max_util = total_util 
            best_pt = pt 
            pt_cost = total_cost 
    print('Items bought:', [student.shopping_list[i] for i, item in enumerate(best_pt) if item == 1])
    print(f'Money spent: {pt_cost} \n"Money saved: {student.budget - pt_cost}\nUtility: {max_util}')
    return best_pt 

'''
For each item, we buy it with probability equal to threshold. 
'''
def sample_point(num_items, threshold=0.7): 
    # does not use previous distribution at all 
    pt = np.zeros((num_items, 1))
    for i in range(num_items): 
        keep_prob = random.random()
        pt[i, :] = 1 if keep_prob < threshold else 0 
    return pt 
        

quadratic_penalty_constraint_method()