from multi_optimization import * 

def graph_hooke_objective_tradeoffs(): 
    random.seed(17)
    np.random.seed(17)
    all_utils, all_saved = [], []
    for p in [0.12, 0.15, 0.17, 0.18, 0.19, 0.2, 0.4, 0.5, 1, 1.2, 1.5, 1.8, 2, 2.3]:
        print(f"P = {p}")
        util, saved = hooke_jeeves_multi(num_its=200, overbudget_penalty=3, spending_penalty=p, ret_values=True)
        all_utils.append(util)
        all_saved.append(saved)
        print(f'Util: {util}, Money saved: {saved}')
    plt.plot(all_utils, all_saved, color='purple')
    plt.grid()
    plt.xlabel("Utility")
    plt.ylabel("Money saved (dollars)")
    plt.title("Hooke Jeeves - Utility vs Money Saved (Varying α)")
    plt.show()


def graph_hooke_budget_analysis(): 
    random.seed(17)
    np.random.seed(17)
    all_utils, all_saved = [], []
    for b in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]:
        print(f"Budget = {b}")
        util, saved = hooke_jeeves_multi(num_its=300, overbudget_penalty=3, spending_penalty=0.18, ret_values=True, budget=b)
        all_utils.append(util)
        all_saved.append(saved)
        print(f'Util: {util}, Money saved: {saved}')
    print(all_utils, all_saved)
    plt.plot(all_utils, all_saved, color='purple')
    plt.grid()
    plt.xlabel("Utility")
    plt.ylabel("Money saved (dollars)")
    plt.title("Hooke Jeeves - Utility vs Money Saved (Varying B)")
    plt.show()

def graph_hooke_pareto_curve(): 
    random.seed(17)
    np.random.seed(17)
    all_utils, all_saved = [], []
    # code pareto frontier points manually from observing the graph because I'm lazy :( 
    pareto_saved = [100, 99, 98, 89, 87, 68, 46, 29, 13]
    pareto_utils = [0, 0.98, 1.88, 7.49, 9.21, 11.81, 15.54, 17.3, 24.64]
    for b in np.linspace(0, 100, 101):
        should_print = True if 100 - b in pareto_saved else False 
        util, saved = hooke_jeeves_multi(num_its=300, overbudget_penalty=1, spending_penalty=0.5, ret_values=True, budget=b, use_pareto=True, should_print=should_print)
        all_utils.append(util)
        all_saved.append(b)
    print(all_utils, all_saved)
    all_saved = 100 - np.array(all_saved)
    plt.scatter(all_utils, all_saved, color='purple')

    plt.grid()
    plt.plot(pareto_utils, pareto_saved, color='navy', label="Pareto frontier")
    plt.xlabel("Utility")
    plt.ylabel("Money saved (dollars)")
    plt.title("Hooke Jeeves - Pareto Frontier")
    plt.legend()
    plt.show()

# graph_hooke_objective_tradeoffs()
# graph_hooke_budget_analysis()
graph_hooke_pareto_curve()