from expression_class import *

import copy
import numpy as np
from collections import deque
from random import randrange
from scipy.optimize import curve_fit



class symbolic_regression():
    
    def __init__(self, config, data_points):
        
        
        self.n_generations         = config["n_generations"]
        self.n_indep_pops          = config["n_indep_pops"]
        self.n_expressions_per_pop = config["n_expressions_per_pop"]
        self.num_mutations         = config["num_mutations"]
        self.p_crossbreed          = config["p_crossbreed"]
        self.tournament_size       = config["tournament_size"]
        
        self.mutation_probs        = config["mutation_probs"]
        
        self.max_complexity        = config["max_complexity"]
        
        self.possible_vars         = config["possible_vars"] # ["a", "b", ...] might need to include "x" !
        
        self.data_points = data_points 
        self.n_data_points = len(self.data_points)
        # list of dicts... [{}, {}, {}]
        # each dict is {"a": a, "b": b, ydata: [y1, y2, y3, y4, ....]}
        
        self.ls = np.array([i for i in range(2, 5000)])
        
        self.func_pops = np.empty((self.n_indep_pops, self.n_expressions_per_pop), dtype=Expression)
        # [i, j] : jth function in population i
        
        self.best_f_per_c_per_p = [{} for k in range(self.n_indep_pops)]
        # for each pop, store dict with best funcs per complexity... best_f_per_c_per_p[13]["5"] : is the best function with complexity 5 in population 13 (14th population)
        
        self.best_f_per_c = {}
        # best_f_per_c["5"] is the best function of complexity 5 seen overall
        
    def init_pops(self):
        
        for pop in range(self.n_indep_pops):
            for i in range(self.n_expressions_per_pop):
                self.func_pops[pop, i] = random_expression(1, self.possible_vars)
                
        print("populations initialized.")
    
    def tournament(self, population):
        
        sub_population = np.random.choice(population, size = self.tournament_size)
        
        fitnesses      = [self.fitness(expression) for expression in sub_population] # DEFINE SELF.FITNESS
        best_ind       = np.argmin(fitnesses)
        
        return sub_population[best_ind]
    
    def evolve_population(self, population):
        
        for i in range(self.num_mutations):
            print(f"mutation: {i+1}")
            if np.random.rand() > self.p_crossbreed:
                # MUTATE
                
                expression = self.tournament(population)
                
                if expression.operation == "const" or expression.operation == "var":
                    action = "new_tree"
                
                action = np.random.choice(["mutate_const", "mutate_form", "mutate_operation", "delete_subtree", "new_tree"], p=self.mutation_probs)
                
                T = 1 - i/self.num_mutations # ANNEALING TEMPERATURE 

                if action == "mutate_const":
                    expression.mutate_random_constant(T) # take input T, use this in the random constant mutation

                elif action == "mutate_form":
                    expression = mutate_expression_form(expression, self.possible_vars)

                elif action == "mutate_operation":
                    expression.mutate_random_operator()

                elif action == "delete_subtree":
                    delete_subtree(expression, self.possible_vars)

                elif action == "new_tree":
                    expression = random_expression(expression.depth, self.possible_vars)
                    
                # ACCEPT MUTATION??
                
            else:
                                        
                parent_1 = self.tournament(population)

                # Make sure both parents are different expressions
                while True:
                    parent_2 = self.tournament(population)

                    if parent_1 != parent_2:
                        break
                        
                # Make sure both kids are not overly complex
                while True:
                    kid_1, kid_2 = self.crossbreed(parent_1, parent_2) # DEFINE

                    if self.check_valid_complexity(kid_1) and self.check_valid_complexity(kid_2):
                        break
                
                replace_ind_1, replace_ind_2 = self.oldest_in_population(population)
                
                population[replace_ind_1] = kid_1
                population[replace_ind_2] = kid_2

    def crossbreed(self, par_1, par_2):
        
        while True:
            
            kid_1 = copy.deepcopy(par_1)
            node_from_par_2 = copy.deepcopy(par_2.select_random_node(select_type = "any"))
            
            while True:
                
                kid_1_mix_at = kid_1.select_random_node(select_type = "any")
                parent = kid_1_mix_at.par
                
                if parent is not None:
                    break

            if parent.left == kid_1_mix_at:
                parent.left = node_from_par_2
                node_from_par_2.par = parent

            elif parent.right == kid_1_mix_at:
                parent.right = node_from_par_2
                node_from_par_2.par = parent
            
            kid_1 = simplify(kid_1)
            
            if not nested_exponent(kid_1):
                break

        while True:
            
            kid_2 = copy.deepcopy(par_2)
            node_from_par_1 = copy.deepcopy(par_1.select_random_node(select_type = "any"))
            
            while True:
                
                kid_2_mix_at = kid_2.select_random_node(select_type = "any")
                parent = kid_2_mix_at.par
                
                if parent is not None:
                    break

            if parent.left == kid_2_mix_at:
                parent.left = node_from_par_1
                node_from_par_1.par = parent

            elif parent.right == kid_2_mix_at:
                parent.right = node_from_par_1
                node_from_par_1.par = parent
            
            kid_2 = simplify(kid_2)
            
            if not nested_exponent(kid_2):
                break
        
        return kid_1, kid_2
    
    def oldest_in_population(self, population):
        
        len_pop = len(population)
        
        oldest1 = population[0]
        oldest2 = population[1]
        
        ind1 = 0
        ind2 = 1
        
        for i in range(2, len_pop):
            if population[i].init_time < oldest1.init_time:
                oldest2 = oldest1
                ind2 = ind1
                
                oldest1 = population[i]
                ind1 = i
            elif population[i].init_time < oldest2.init_time:
                oldest2 = population[i]
                ind2 = i
        
        return ind1, ind2
    
    
    def do_symbolic_regression(self):
        
        self.init_pops()
        
        for generation in range(1, self.n_generations + 1):
            print("current generation:", generation)
            for population in range(self.n_indep_pops):
                
                print(f"evolving population#{population + 1}.")
                # Mutate & Crossbreed through the entire population
                self.evolve_population(self.func_pops[population]) # DEFINE
                
                print(f"optimizing and simplifying functions in #{population + 1}.")
                for expression in self.func_pops[population]:
                    expression = self.optimize(expression) # DEFINE
                    expression = simplify(expression) # DEFINE
                    
                    c = expression.get_complexity()
                    fitness = self.fitness(expression) # DEFINE
                    
                    
                    # update best_f_per_c_per_p
                    if c not in self.best_f_per_c_per_p[population]:
                        self.best_f_per_c_per_p[population][c] = expression
                    elif fitness < self.fitness(self.best_f_per_c_per_p[population][c]):
                        self.best_f_per_c_per_p[population][c] = expression
                    
                    # update best_f_per_c
                    if c not in self.best_f_per_c:
                        self.best_f_per_c[c] = expression
                    elif fitness < self.fitness(self.best_f_per_c[c]):
                        self.best_f_per_c[c] = expression
                
                #for func in population:
                    # if rand() < merge_prob_1:
                    # func = random func from best_f_per_c_per_p[population]
                    
                    # if rand() < merge_prob_2:
                    # func = random func from best_f_per_c
        
        return self.best_f_per_c
    
    def eval_l(self, expression, ell, data_point, ell_c = 4000, ell_d = 1200, m = 1.2):
        
        wl = 1/(1+np.exp(-(ell-ell_c)/100))
        
        alpha = expression.compute(data_point)
        
        val = 1 - wl + wl*((ell/ell_d)**alpha)/np.exp(-(ell/ell_d)**m)
        
        return val
        
    
    def fitness(self, expression):
        
        n_data_points = 5
        optimization_data_points = np.random.choice(self.data_points, size = n_data_points) # optimize using three data points
        
        fitness = 0
        
        for data_point in optimization_data_points:
            for i, ell in enumerate(self.ls):

                true = data_point["ydata"][i]
                pred = self.eval_l(expression, ell, data_point)

                fitness += np.abs((true - pred)/(true))
            
        fitness /= n_data_points
        
        return fitness
    
    def optimize(self, expression):
        
        n_data_points = 5
        optimization_data_points = np.random.choice(self.data_points, size = n_data_points) # optimize using three data points
        
        copied_e = copy.deepcopy(expression)
        consts = []
        d = deque([copied_e])
        
        # search E, create an array which points to all the (tunable) constants in E
        
        while len(d) > 0:
            cur = d.pop()
            
            if cur.operation == "const":
                consts.append(cur)
            elif cur.operation != "var":
                d.append(cur.left)
                d.append(cur.right)
                
        num_consts = len(consts)
        
        # initial guess for optimization is whatever the constants already are
        p0 = [const.left for const in consts]
        
        # array to store the optimal constants learned using each datapoint
        optim_consts = np.zeros((n_data_points, num_consts))
        
        for i, data_point in enumerate(optimization_data_points):
        
            def func(x, *args):

                for i, arg in enumerate(args):
                    consts[i].left = arg

                return self.eval_l(copied_e, x, data_point)
            
            try:
                popt, _ = curve_fit(func, self.ls, data_point["ydata"], p0 = p0)
            except: 
                popt = p0

            optim_consts[i] = np.real(popt)
            
        for k, const in enumerate(consts):
            const.left = np.random.choice(optim_consts[:, k])
        
        return copied_e

    def check_valid_complexity(self, expression):
        if expression.get_complexity() > self.max_complexity:
            return False
        return True

    