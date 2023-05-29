""" 
If we make this problem being able to buy multiple of each item (constrained by max)
it's multiobjective multidimensional optimization problem except with 
linear constraints. 
"""
import random 
import numpy as np 
from CollegeStudent import CollegeStudent
from utils import *
import matplotlib.pyplot as plt 

'''
Note that in this method, we can buy up to the item limit. 
With probability threshold, we do not buy the item. 
'''
def sample_point_multi(student: CollegeStudent, saving_threshold=0.2): 
    # does not use previous distribution at all 
    pt = np.zeros((len(student.shopping_list), 1))
    for i in range(len(student.shopping_list)): 
        pt[i] = random.choice(range(int(student.max_amounts[i])+1))
        if random.random() < saving_threshold: 
            pt[i] = 0
    return pt 

'''
Quadratic penalty function with the possibility of buying multiple of each item,
up to a maximum amount specified in the shopping list. 
'''
def quad_penalty_multi(file='shopping_list.txt', num_its=20000, overbudget_penalty=1, spending_penalty=1, plot=False): 
    student = CollegeStudent(file)
    max_util_with_penalty = float('-inf')
    max_util = None 
    pt_cost = None 
    best_pt = None 
    if plot: history = []
    for it in range(num_its): 
        pt = sample_point_multi(student, saving_threshold=0.15)
        total_util, total_cost = calc_util_and_cost(pt, student.shopping_utils, student.shopping_costs)
        util_plus_penalty = calc_util_plus_penalty(total_util, total_cost, overbudget_penalty, spending_penalty, student.budget)
        if util_plus_penalty > max_util_with_penalty: 
            max_util_with_penalty = util_plus_penalty
            max_util = total_util 
            best_pt = pt 
            pt_cost = total_cost 
            # history.append(max_util_with_penalty)
        if plot and it % 1000 == 0: history.append(util_plus_penalty)
    print_results(student, best_pt, pt_cost, max_util)
    if plot: 
        plt.plot([i+1 for i in range(len(history))], history)
        plt.show()
    return best_pt 


'''
Hooke Jeeves for multi optimization. 
We manually constrain the problem to prevent Hooke Jeeves from moving outside the constraint boundaries. 
We also don't shrink the step size - every step is just 1. 
'''
def hooke_jeeves_multi(file='shopping_list.txt', num_its=60, overbudget_penalty=0.05, spending_penalty=0.05, plot=False): 
    student = CollegeStudent(file)
    shopping_list_len = len(student.shopping_list)
    best_pt = sample_point_multi(student)
    total_util, total_cost = calc_util_and_cost(best_pt, student.shopping_utils, student.shopping_costs)
    util_plus_penalty = calc_util_plus_penalty(total_util, total_cost, overbudget_penalty, spending_penalty, student.budget)
    if plot: history = []
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
        if plot: history.append(best_util_plus_penalty)
        # only update everything after exploring all dimensions in an iteration 
        total_util, total_cost = best_util, best_cost
        util_plus_penalty = best_util_plus_penalty
        best_pt = best_pt_after_it
    if plot: 
        plt.plot([i+1 for i in range(len(history))], history)
        plt.show()
    print_results(student, best_pt, total_cost, total_util)

def transition_to_new_pt(old_pt, student: CollegeStudent, threshold=0.3): 
    new_pt = np.zeros(old_pt.shape)
    for i in range(len(student.shopping_list)):
        if random.random() < threshold: 
            new_pt[i] = random.choice(range(int(student.max_amounts[i])+1)) # satisfy the constraints 
        else:
            new_pt[i] = old_pt[i]
    return new_pt 

'''
Simulated annealing algorithm. 
'''
def simulated_annealing_multi(file='shopping_list.txt', num_its=200, overbudget_penalty=0.5, spending_penalty=0.5, temp=5, decay=1.01, plot=False):
    student = CollegeStudent(file) 
    best_pt = sample_point_multi(student)
    total_util, total_cost = calc_util_and_cost(best_pt, student.shopping_utils, student.shopping_costs)
    best_util_plus_penalty = calc_util_plus_penalty(total_util, total_cost, overbudget_penalty, spending_penalty, student.budget)
    cur_pt = best_pt
    cur_util_plus_penalty = best_util_plus_penalty
    if plot: history = [cur_util_plus_penalty]
    for it in range(num_its): 
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
        if plot: history.append(new_util_plus_penalty)
    if plot: 
        plt.plot([i+1 for i in range(len(history))], history)
        plt.show()
    print_results(student, best_pt, total_cost, total_util)


def genetic_algorithm_multi(file='shopping_list.txt', num_its=100, overbudget_penalty=0.5, spending_penalty=0.5, pop_size=2000, mutation_rate=0.5, plot=False): 
    # TODO: include visuals of how the chromosomes get recombined 
    student = CollegeStudent(file)

    # sample initial population 
    pop = [sample_point_multi(student) for _ in range(pop_size)] 
    best_pt = None 
    total_cost, total_util = None, None 

    for _ in range(num_its):
        new_pop = []
        all_util_costs = [calc_util_and_cost(pt, student.shopping_utils, student.shopping_costs) for pt in pop]
        obj_evals = np.array([calc_util_plus_penalty(all_util_costs[i][0], all_util_costs[i][1], overbudget_penalty, spending_penalty, student.budget) for i in range(len(all_util_costs))])
        
        # select top k parents every time  
        # reference: https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        k = len(pop) // 2
        k = k - 1 if k % 2 != 0 else k # make k even, round down  
        if k <= 1: break # break early if no parent pairs can be formed 
        top_k_indices = np.argpartition(obj_evals, -k)[-k:]
        top_k_indivs = [pop[i] for i in top_k_indices]

        best_pt = top_k_indivs[-1]
        total_util, total_cost = all_util_costs[top_k_indices[-1]]

        # uniform crossover
        for p in range(0, k, 2): 
            parent1, parent2 = top_k_indivs[p], top_k_indivs[p+1]
            child = np.array([parent1[b] if random.random() < 0.5 else parent2[b] for b in range(len(parent1))]).reshape(-1, 1)
            new_pop.append(child)
            # mutate child 
            if random.random() < mutation_rate: 
                child = transition_to_new_pt(child, student, threshold=1/len(child)) # rate is 1/m 
        pop = new_pop 
    
    print_results(student, best_pt, total_cost, total_util)
