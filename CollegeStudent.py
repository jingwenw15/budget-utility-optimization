import numpy as np


class CollegeStudent: 
    def __init__(self, budget): 
        self.budget = budget
        self.shopping_list, self.shopping_utils = self.extract_list_and_utils()
        print(self.shopping_utils, self.shopping_list)

    def extract_list_and_utils(self): 
        file = open('shopping_list.txt', 'r')
        items_and_utils = file.read().split('\n')
        shopping_list, shopping_utils = [], []
        for item_and_util in items_and_utils: 
            item, util = item_and_util.split(',')
            shopping_list.append(item.strip())
            shopping_utils.append(float(util.strip()))

        return shopping_list, np.array(shopping_utils)

