from CollegeStudent import CollegeStudent
import numpy as np 
import random 


def penalty_method(num_its=20, penalty=500): 
    # first assume that you want to spend the entire budget 
    student = CollegeStudent()
    max_util = float('-inf')
    pt_cost = None 
    best_pt = None 
    for _ in range(num_its): 
        pt = sample_point(len(student.shopping_list))
        total_util = np.sum(pt * student.shopping_utils) 
        total_cost = np.sum(pt * student.shopping_costs)
        util_plus_penalty = total_util - penalty * max(total_cost - student.budget, 0)
        if util_plus_penalty > max_util: 
            max_util = util_plus_penalty
            best_pt = pt 
            pt_cost = total_cost 
    print([student.shopping_list[i] for i, item in enumerate(best_pt) if item == 1])
    print(f'Money spent: {pt_cost} \nUtility: {max_util}')
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
        

penalty_method()