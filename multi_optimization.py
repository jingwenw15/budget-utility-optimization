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
We randomly sample points. 
'''
def quad_penalty_sampling_multi(file='shopping_list.txt', num_its=20000, overbudget_penalty=0.05, spending_penalty=0.05, plot=False): 
    student = CollegeStudent(file)
    max_util_with_penalty = float('-inf')
    max_util = None 
    pt_cost = None 
    best_pt = None 
    if plot: history = []
    for it in range(num_its): 
        pt = sample_point_multi(student, saving_threshold=0.9)
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
def hooke_jeeves_multi(file='shopping_list.txt', num_its=100, overbudget_penalty=0.05, spending_penalty=0.07, plot=False, ret_values=False, budget=100): 
    student = CollegeStudent(file, budget=budget)
    shopping_list_len = len(student.shopping_list)
    best_pt = sample_point_multi(student)
    total_util, total_cost = calc_util_and_cost(best_pt, student.shopping_utils, student.shopping_costs)
    util_plus_penalty = calc_util_plus_penalty(total_util, total_cost, overbudget_penalty, spending_penalty, student.budget)
    if plot: 
        best_util_plus_penalty_history = []
        best_util_history = []
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
        if plot: 
                    best_util_plus_penalty_history.append(best_util_plus_penalty)
                    best_util_history.append(best_util)
        # only update everything after exploring all dimensions in an iteration 
        total_util, total_cost = best_util, best_cost
        util_plus_penalty = best_util_plus_penalty
        best_pt = best_pt_after_it
    if plot: 
        plt.plot([i+1 for i in range(len(best_util_plus_penalty_history))], best_util_plus_penalty_history, color='navy', label="Best utility plus penalty")
        plt.plot([i+1 for i in range(len(best_util_history))], best_util_history, color='violet', label="Best utility")
        plt.xlabel("Number of iterations")
        plt.ylabel("Best objective value at each iteration")
        plt.title(f"Hooke Jeeves - Multi-Value Optimization, β={0.2}, ρ={overbudget_penalty}, α={spending_penalty}")
        plt.xlim(0, 60)
        plt.ylim(-40, 40)
        plt.grid()
        plt.legend()
        plt.show()
    if not ret_values: print_results(student, best_pt, total_cost, total_util)
    else: return total_util, 100 - total_cost

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
def simulated_annealing_multi(file='shopping_list.txt', num_its=90, overbudget_penalty=0.05, spending_penalty=0.1, temp=5, decay=0.95, plot=False):
    student = CollegeStudent(file) 
    best_pt = sample_point_multi(student)
    total_util, total_cost = calc_util_and_cost(best_pt, student.shopping_utils, student.shopping_costs)
    best_util_plus_penalty = calc_util_plus_penalty(total_util, total_cost, overbudget_penalty, spending_penalty, student.budget)
    cur_pt = best_pt
    cur_util_plus_penalty = best_util_plus_penalty
    best_util = total_util
    if plot: 
        cur_util_plus_penalty_history = [cur_util_plus_penalty]
        cur_util_history = [total_util]
    for it in range(num_its): 
        new_pt = transition_to_new_pt(cur_pt, student)
        new_util, new_cost = calc_util_and_cost(new_pt, student.shopping_utils, student.shopping_costs)
        new_util_plus_penalty = calc_util_plus_penalty(new_util, new_cost, overbudget_penalty, spending_penalty, student.budget)
        if new_util_plus_penalty > cur_util_plus_penalty or random.random() < np.exp((new_util_plus_penalty - cur_util_plus_penalty)/(temp * decay) + 1e-8):
            cur_pt = new_pt 
            cur_util_plus_penalty = new_util_plus_penalty
        if new_util_plus_penalty > best_util_plus_penalty:  # global update 
            best_util_plus_penalty = new_util_plus_penalty
            best_pt = new_pt 
            total_util, total_cost = new_util, new_cost 
        if new_util > best_util: best_util = new_util
        if plot: 
            cur_util_plus_penalty_history.append(best_util_plus_penalty)
            cur_util_history.append(best_util)
    if plot: 
        plt.plot([i+1 for i in range(len(cur_util_plus_penalty_history))], cur_util_plus_penalty_history, color='navy', label='Best utility plus penalty')
        plt.plot([i+1 for i in range(len(cur_util_history))], cur_util_history, color='violet', label="Best utility")
        plt.xlabel("Number of iterations")
        plt.ylabel("Best objective value at each iteration")
        plt.title(f"Simulated Annealing - Multi-Value Optimization, β={0.2}, ρ={overbudget_penalty}, α={spending_penalty}")
        plt.xlim(0, 90)
        plt.grid()
        plt.legend()
        plt.show()
    print_results(student, best_pt, total_cost, total_util)


def genetic_algorithm_multi(file='shopping_list.txt', num_its=500, overbudget_penalty=0.05, spending_penalty=0.4, pop_size=10000, mutation_rate=0.3, plot=False): 
    student = CollegeStudent(file)

    # sample initial population 
    pop = [sample_point_multi(student, saving_threshold=0.5) for _ in range(pop_size)] 
    best_pt = None 
    total_cost, total_util = None, None 

    for it in range(num_its):
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
            # mutate child 
            if random.random() < mutation_rate: 
                child = transition_to_new_pt(child, student, threshold=1/len(child)) # rate is 1/m 
            new_pop.append(child)
        pop = new_pop 
    
    print_results(student, best_pt, total_cost, total_util)
