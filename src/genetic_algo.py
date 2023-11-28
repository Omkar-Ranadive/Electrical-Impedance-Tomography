import numpy as np 
from utils import cal_vol


class Genetic(): 
    def __init__(self, arr, num_generations, population_size, num_parents, mutation_rate, print_every=None) -> None:
        """

        Args:
            arr (np.ndarray): Input array containing the data 
            num_generations (int): Amount of iterations to run the generation algorithm 
            population_size (int): Size of population. For example, if population_size=100, original data arr is 
                                    (378, 26) then a matrix of (100, 26) gets created where each row is 
                                   one "population", i.e, randomly chosen row from original matrix (size 378)
            num_parents (int): Number of parents to keep in next generation after fitness test 
            mutation_rate (float): Children wil randomly be mutated with mutation_rate  
            print_every (int, optional): If provide, print the best results every print_every iterations. 
                                         Defaults to None.
        """
        self.arr = arr 
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_parents = num_parents
        assert self.population_size >= self.num_parents, "Num parents can't be greater than total population"
        self.num_children = self.population_size - self.num_parents
        self.mutation_rate = mutation_rate
        self.num_rows, self.num_cols = self.arr.shape[0], self.arr.shape[1] 
        self.print_every = print_every
    
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
    

    def _select_parents(self, fitness_values):
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
                print("*"*100)
                            
            fitness_scores = self._fitness() 
            selected_parents = self._select_parents(fitness_scores)

            max_fitness = cal_vol(self.arr[selected_parents[-1]])
            if max_fitness > best_dict['best_score']: 
                best_dict['best_score'] = max_fitness 
                best_dict['best_indices'] = selected_parents[-1].copy()
                best_dict['best_generation'] = generation

            children = self._crossover(selected_parents)
            children = self._mutate(children) 
            new_population = np.concatenate((selected_parents, children)) 
            self.population = new_population 

        return best_dict 








