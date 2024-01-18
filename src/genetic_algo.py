import numpy as np 
from utils_math import cal_vol, get_norm_with_rank
from itertools import combinations
import algos 
# from numba import jit 


class Genetic(): 
    def __init__(self, arr, num_generations, population_size, 
                num_parents, mutation_rate, num_entries, 
                random_children_rate, direct_mutants_rate, adaptive_mutation, 
                adaptive_increase=1.2, adaptive_decrease=0.8, improvement_check=50,
                print_every=None) -> None:
        """
        Args:
            arr (np.ndarray): Input array containing the data 
            num_generations (int): Amount of iterations to run the generation algorithm 
            population_size (int): Size of population. For example, if population_size=100, original data arr is 
                                    (378, 26) then a matrix of (100, 26) gets created where each row is 
                                   one "population", i.e, randomly chosen row from original matrix (size 378)
            num_parents (int): Number of parents to keep in next generation after fitness test 
            mutation_rate (float): Children wil randomly be mutated with mutation_rate  
            improvement_check (int): If fitness doesn't increase every improvement_check generation, 
                                     increase mutation rate
            random_children_rate (float): At each generation, this percent of population will be randomly generated.
            direct_mutants_rate (float): At each generation, the best parent will be mutated rate*population_size.  
            num_entries (int): Number of features/measurements, i.e, the total number of indices which get selected 

            adaptive_mutation (bool): If true, mutation rate adapts to fitness over generations
            adaptive_increase (float): Rate by which to increase mutation rate (only used if adaptive_mutation=True)
            adaptive_decrease (float): Rate by which to decrease mutation rate (only used if adaptive_mutation=True)

            print_every (int, optional): If provide, print the best results every print_every iterations. 
                                         Defaults to None.
        """
        self.arr = arr 
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_parents = num_parents
        assert self.population_size >= self.num_parents, "Num parents can't be greater than total population"
        self.random_children = int(random_children_rate * population_size)  
        self.num_direct_mutants = int(direct_mutants_rate * population_size)
        self.num_children = self.population_size - self.num_parents - self.random_children - self.num_direct_mutants
        self.mutation_rate = mutation_rate
        self.original_rate = mutation_rate
        self.num_rows, self.num_cols = self.arr.shape[0], num_entries
        self.adaptive_mutation = adaptive_mutation
        self.print_every = print_every
        self.improvement_check = improvement_check 
        self.population = np.empty((population_size, self.num_cols), dtype=int)

        for i in range(self.population_size): 
            self.population[i] = np.random.choice(self.num_rows, self.num_cols, replace=False)


    def _fitness(self): 
        """
        Calculate fitness of the population. Fitness score = volume 

        Returns:
            np.ndarray: Fitness scores, i.e, the volumes for each population entry 
        """
        return np.array([cal_vol(self.arr[indices]) for indices in self.population])
    

    def _select_parents(self, fitness_values: np.ndarray) -> np.ndarray:
        """
        Select the most fit self.num_parents number of parents 
        Args:
            fitness_values (np.ndarray): Numpy array containing the fitness values (volume) for the population

        Returns:
            np.ndarray: Numpy array with the most fit population  
        """

        selected_indices = np.argsort(fitness_values)[-self.num_parents:]
        return self.population[selected_indices]


    def _crossover(self, parents):
        """
        Create children using the most fit parents 
        Args:
            parents (np.ndarray): Numpy array of the most fit parents 

        Returns:
            np.ndarray: Children formed using the most fit parents 
        """

        children = []

        # Randomly form children from all parents 
        for _ in range(self.num_children):
            p1_index, p2_index = np.random.choice(len(parents), 2, replace=False)
            parent1, parent2 = parents[p1_index], parents[p2_index]
            crossover_point = np.random.randint(self.num_cols)

            # The entries (col indices) need to be unique for every child. 
            remaining_points = self.num_cols - crossover_point 
            gene1 = parent1[:crossover_point] 
            gene2 = np.setdiff1d(parent2, gene1)[:remaining_points]

            child = np.concatenate((gene1, gene2))
            children.append(child)

        return np.array(children)


    def _mutate(self, children):
        """
        Mutate an entry (column index) within a child with a small probability 
        Args:
            children (np.ndarray): Numpy array of children 

        Returns:
            np.ndarray: Numpy array of mutated children 
        """

        for child in children: 
            if np.random.uniform() < self.mutation_rate: 
                mutation_col = np.random.randint(self.num_cols) 
                available_indices = np.setdiff1d(np.arange(self.num_rows), child)
                child[mutation_col] = int(np.random.choice(available_indices))
            
        return children 


    def _get_random_children(self): 
        """
        Generate a set of random children during each generation. 
        Number of randomly generated children = random_children_rate * population_size

        Returns:
            np.ndarray: Randomly generated children 
        """
        children = np.empty((self.random_children, self.num_cols), dtype=int)

        for i in range(self.random_children): 
            children[i] = np.random.choice(self.num_rows, self.num_cols, replace=False)
        
        return children 


    def _get_direct_mutants(self, candidate): 
        """
        Directly mutate the best candidate at each generation 
        Args:
            candidate (np.ndarray): The candidate producing the highest fitness (volume)

        Returns:
            np.ndarray: Mutated versions of the best candidate 
        """
        mutants = np.empty((self.num_direct_mutants, self.num_cols), dtype=int)

        for i in range(self.num_direct_mutants): 
            mutants[i] = candidate.copy()
            mutation_col = np.random.randint(self.num_cols) 
            available_indices = np.setdiff1d(np.arange(self.num_rows), candidate)
            mutants[i][mutation_col] = int(np.random.choice(available_indices))
        
        return mutants


    def run_genetic_simulation(self):
        """
        Run genetic algorithm for num_generations and return the best results obtained 

        Returns:
            dict: Dictionary containing the best values of score, indices, generation 
        """

        best_dict = {'best_score': -np.inf, 'best_indices': None, 'best_generation': -1}

        for generation in range(self.num_generations): 
            if self.print_every and generation % self.print_every == 0: 
                print(f"Running for generation: {generation}")
                print(f"Best Volume till now: {best_dict['best_score']}")
                print(f"Selected Indices: {best_dict['best_indices']}")
                print(f"Mutation Rate: {self.mutation_rate}")
                print("*"*100)

            fitness_scores = self._fitness()
            selected_parents = self._select_parents(fitness_scores)

            max_fitness = cal_vol(self.arr[selected_parents[-1]])
            if max_fitness > best_dict['best_score']: 
                best_dict['best_score'] = max_fitness 
                best_dict['best_indices'] = selected_parents[-1].copy()
                best_dict['best_generation'] = generation

                if self.adaptive_mutation: 
                    # If improvement is observed, reduce mutation rate
                    self.mutation_rate = max(self.original_rate, self.mutation_rate*0.8) 

            children = self._crossover(selected_parents)
            children = self._mutate(children) 
            random_children = self._get_random_children()
            direct_mutants = self._get_direct_mutants(selected_parents[-1])
            new_population = np.concatenate((selected_parents, children, random_children, direct_mutants)) 
            self.population = new_population 

            if self.adaptive_mutation and generation % self.improvement_check == 0: 
                # If no improvement over k generations, increase mutation rate
                if generation - best_dict['best_generation'] > self.improvement_check: 
                    self.mutation_rate = min(self.mutation_rate*1.2, 1.0) 


        best_dict['best_indices'] = sorted(best_dict['best_indices'])
        return best_dict 


class EnsembleGenetic(Genetic):
    def __init__(self, gen_instances, gen_dict_instances, gen_dict_ensemble, logger=None) -> None:
        """
        Ensemble Genetic which runs gen_instances number of genetic algorithms, collects the results, and runs a final
        genetic algorithm with these collected results as starting population (along with other random population). 
        Args:
            gen_instances (int): Amount of instances of genetic algorithm to run in the first phase 
            gen_dict_instances (dict): Parameters used by the genetic algorithm in the first phase 
            gen_dict_ensemble (_type_): Parameters used by the genetic algorithm in the final phase (ensemble step)
            logger (_type_, optional): If a logger instance is provided, information is logged to a file. 
                                       Defaults to None.
        """

        self.gen_instances = gen_instances 
        self.gen_dict_instances = gen_dict_instances
        self.candidates = []
        self.scores = []
        self.logger = logger
        super().__init__(**gen_dict_ensemble)


    def _print(self, output):
        if self.logger is not None: 
            print(output) 
            self.logger.info(output)
        else: 
            print(output)
        

    def run_ensemble_genetics(self): 
        early_dict = {'best_score': -np.inf, 'best_indices': None, 'best_generation': -1}
        early_solution = False 
        unique_candidates = set()


        # Run n runs of genetic algorithm over a small set of iterations to get "good" candidates 
        for i in range(self.gen_instances):
            # Terminate early if sol space is too small  
            if i == 5 and len(unique_candidates) == 1:
                early_solution = True 
                break 

            self._print(f"Running genetic instance: {i}")
            GenObj = Genetic(**self.gen_dict_instances) 
            best_results = GenObj.run_genetic_simulation()
            self.candidates.append(best_results['best_indices'])
            self.scores.append(best_results['best_score'])
            unique_candidates.add(tuple(best_results['best_indices']))

            self._print(f"Volume: {best_results['best_score']}")
            self._print(f"Selected Indices: {best_results['best_indices']}")
            self._print("*"*20)


        # Run an instance of greedy approach 
        greedy_vol, greedy_indices = algos.greedy_approach(self.arr, self.num_cols)
        self.candidates.append(greedy_indices) 
        self.scores.append(greedy_vol)

        # Run an instance of highest magnitude approach 
        magnitudes, sorted_indices = get_norm_with_rank(self.arr) 
        self.candidates.append(sorted_indices[:self.num_cols])
        self.scores.append(cal_vol(self.arr[self.candidates[-1]]))

        self.candidates = np.array(self.candidates)
        self._print("Best candidate obtained in first phase: ")
        best_first_phase = np.argmax(self.scores)
        self._print(f'Volume: {self.scores[best_first_phase]}')
        self._print(f'Indices: {self.candidates[best_first_phase]}')
        early_dict['best_score'] = self.scores[best_first_phase]
        early_dict['best_indices'] = self.candidates[best_first_phase]

        if not early_solution:
            # Add these candidates to the initial population 
            self.population = np.concatenate((self.population, self.candidates)) 
            # Run the genetic algorithn again with "good" candidates mixed with the population 
            return self.run_genetic_simulation()
        else: 
            return early_dict
        

class GreedyGenetic(Genetic):
    def __init__(self, gen_dict, greedy_it, num_greedy):
        super().__init__(**gen_dict)

        # Get the magnitudes for all rows 
        self.mag, self.sorted_mag_indices = get_norm_with_rank(self.arr)

        self.greedy_it = greedy_it 
        self.num_greedy = num_greedy


    def _greedy_selection(self, candidate): 
        # Randomly choose indices to remove 
        indices_to_remove = np.random.choice(np.arange(self.num_cols), size=self.num_greedy, replace=False)

        # Get remaining indices, i.e, the ones not part of the current candidate 
        remaining_indices = np.setdiff1d(np.arange(self.num_rows), candidate)
        rindices_count = len(remaining_indices)
        og_vol = cal_vol(self.arr[candidate])

        new_candidates = np.empty((self.num_greedy * rindices_count, self.num_cols), dtype=int)
        new_volumes = np.empty((self.num_greedy * rindices_count))

        for rindex in range(self.num_greedy): 
            new_candidate = np.delete(candidate, indices_to_remove[rindex])        
            for i, row_index in enumerate(remaining_indices): 
                new_candidate_i = np.concatenate((new_candidate, [row_index]))
                new_candidates[rindex * rindices_count + i] = new_candidate_i
                new_volumes[rindex * rindices_count + i] = cal_vol(self.arr[new_candidate_i])
             
        # Selected candidates are those which have a higher volume than original candidate 
        sorted_volumes = np.argsort(new_volumes)[::-1]
        selected_candidates = new_candidates[sorted_volumes]

        if len(selected_candidates) >= self.num_greedy*self.num_greedy: 
            return selected_candidates[:self.num_greedy*self.num_greedy]
        else: 
            # If enough selected candidates do not exist, then randomly mutate the rest 
            random_mutants = self.num_greedy*self.num_greedy - len(selected_candidates)
        
            mutants = np.empty((random_mutants, self.num_cols), dtype=int)
            for i in range(random_mutants): 
                mutants[i] = candidate.copy()
                mutation_col = np.random.randint(self.num_cols) 
                available_indices = np.setdiff1d(np.arange(self.num_rows), candidate)
                mutants[i][mutation_col] = int(np.random.choice(available_indices))
            
            return np.concatenate((selected_candidates, mutants))
            

    def run_genetic_simulation(self):
        """
        Run genetic algorithm for num_generations and return the best results obtained 

        Returns:
            dict: Dictionary containing the best values of score, indices, generation 
        """

        best_dict = {'best_score': -np.inf, 'best_indices': None, 'best_generation': -1}

        for generation in range(self.num_generations): 
            if self.print_every and generation % self.print_every == 0: 
                print(f"Running for generation: {generation}")
                print(f"Best Volume till now: {best_dict['best_score']}")
                print(f"Selected Indices: {best_dict['best_indices']}")
                print(f"Mutation Rate: {self.mutation_rate}")
                print("*"*100)

            fitness_scores = self._fitness()
            selected_parents = self._select_parents(fitness_scores)

            max_fitness = cal_vol(self.arr[selected_parents[-1]])
            if max_fitness > best_dict['best_score']: 
                best_dict['best_score'] = max_fitness 
                best_dict['best_indices'] = selected_parents[-1].copy()
                best_dict['best_generation'] = generation

                if self.adaptive_mutation: 
                    # If improvement is observed, reduce mutation rate
                    self.mutation_rate = max(self.original_rate, self.mutation_rate*0.8) 

            children = self._crossover(selected_parents)
            children = self._mutate(children) 
            random_children = self._get_random_children()
            direct_mutants = self._get_direct_mutants(selected_parents[-1])
            new_population = np.concatenate((selected_parents, children, random_children, direct_mutants)) 

            if generation % self.greedy_it == 0: 
                all_greedy_candidates = []
                for candidate in selected_parents:
                    greedy_candidates = self._greedy_selection(candidate)
                    all_greedy_candidates.append(greedy_candidates)
                        
                all_greedy_candidates = np.array(all_greedy_candidates).reshape(-1, self.num_cols)
                print(f"Best greedy solution at it {generation}: {cal_vol(self.arr[all_greedy_candidates[0]])}")
                new_population = np.concatenate((new_population, all_greedy_candidates))

            self.population = new_population 

            if self.adaptive_mutation and generation % self.improvement_check == 0: 
                # If no improvement over k generations, increase mutation rate
                if generation - best_dict['best_generation'] > self.improvement_check: 
                    self.mutation_rate = min(self.mutation_rate*1.2, 1.0) 


        best_dict['best_indices'] = sorted(best_dict['best_indices'])
        return best_dict 


    