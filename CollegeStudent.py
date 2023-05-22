import numpy as np

# TODO: later on, allow user to specify path to their own shopping list 

class CollegeStudent: 
    def __init__(self, file="shopping_list.txt", budget=100): 
        self.budget = budget
        self.shopping_list, self.shopping_utils, self.shopping_costs, self.max_amounts = self.extract_txt_file(file)
        self.shopping_utils = self.shopping_utils.reshape((self.shopping_utils.shape[0], 1))
        self.shopping_costs = self.shopping_costs.reshape((self.shopping_costs.shape[0], 1))
        self.max_amounts = self.max_amounts.reshape((self.max_amounts.shape[0], 1))


    def extract_txt_file(self, file="shopping_list.txt"): 
        file = open(file, 'r')
        items_and_utils = file.read().split('\n')
        shopping_list, shopping_utils, shopping_costs, max_amounts = [], [], [], []
        for item_and_util in items_and_utils: 
            item, util, cost, amount = item_and_util.split(',')
            shopping_list.append(item.strip())
            shopping_utils.append(float(util.strip()))
            shopping_costs.append(float(cost.strip()))
            max_amounts.append(float(amount.strip()))

        return shopping_list, np.array(shopping_utils), np.array(shopping_costs), np.array(max_amounts)

