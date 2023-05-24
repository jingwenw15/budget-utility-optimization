import numpy as np


def calc_util_and_cost(pt, shopping_utils, shopping_costs): 
    return np.sum(pt * shopping_utils), np.sum(pt * shopping_costs)

def calc_util_plus_penalty(total_util, total_cost, overbudget_penalty, spending_penalty, budget):
    return total_util - overbudget_penalty * (max(total_cost - budget, 0) ** 2) - ((spending_penalty * total_cost) ** 2)


def print_results(student, best_pt, pt_cost, max_util):
    print('Items bought:', [student.shopping_list[i] for i, item in enumerate(best_pt) if item > 0])
    print('Quantities of each item', [item[0] for item in best_pt if item > 0])
    print(f'Money spent: {pt_cost} \n"Money saved: {student.budget - pt_cost}\nUtility: {max_util}')
