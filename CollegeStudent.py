import numpy as np


class CollegeStudent: 
    def __init__(self, budget=100): 
        self.budget = budget
        self.shopping_list, self.shopping_utils, self.shopping_costs = self.extract_list_and_others()
        self.shopping_utils = self.shopping_utils.reshape((self.shopping_utils.shape[0], 1))
        self.shopping_costs = self.shopping_costs.reshape((self.shopping_costs.shape[0], 1))

    def extract_list_and_others(self): 
        file = open('shopping_list.txt', 'r')
        items_and_utils = file.read().split('\n')
        shopping_list, shopping_utils, shopping_costs = [], [], []
        for item_and_util in items_and_utils: 
            item, util, cost = item_and_util.split(',')
            shopping_list.append(item.strip())
            shopping_utils.append(float(util.strip()))
            shopping_costs.append(float(cost.strip()))

        return shopping_list, np.array(shopping_utils), np.array(shopping_costs)

