from algo_genetic import GreedyGenetic
from constants import DATA_PATH, EXP_PATH, data_list
import time 
import utils_math
import scipy.io as sio
import logging 
import os 
from datetime import datetime
import argparse 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=0, type=int, help="Starting index of data list")
    parser.add_argument("--end", default=len(data_list), type=int, help="Ending index of data list")
    parser.add_argument("--greedy_it", default=30, type=int, help="Run greedy algo every greedy_it in between genetic")
    parser.add_argument("--num_greedy", default=10, type=int, help="Number of greedy solutions to choose")

    # Genetic algorithm dictionary 
    parser.add_argument("--gi_gen", default=1000, type=int, help="Number of generations")
    parser.add_argument("--gi_pop", default=2500, type=int, help="Population size")
    parser.add_argument("--gi_par", default=200, type=int, help="Number of parents")
    parser.add_argument("--gi_mr", default=0.2, type=float, help="Mutation rate")
    parser.add_argument("--gi_cr", default=0.2, type=float, help="Random children rate for each generation")
    parser.add_argument("--gi_dmr", default=0.05, type=float, help="Direct mutation rate for each generation")

    args = parser.parse_args()

    # Load the data 
    for mat_info in data_list[args.start:args.end]: 
        C, M, D = mat_info
        print(f"Running for: {C}contacts_{M}polys_D{D}")
        data = sio.loadmat(DATA_PATH / 'OptimizationMatrices' / f'{C}contacts_{M}polys.mat')
        arr = data['JGQ']
        EXP_DIR = EXP_PATH / 'OptimizationMatrices_GreedyGen' / f'{C}contacts_{M}polys_D{D}'
        os.makedirs(EXP_DIR, exist_ok=True)

        # Set up logger 
        logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / 'info.log'), format='%(message)s')
        logger=logging.getLogger() 
        time_stamp = datetime.now()
        logger.info(f"Running for Matrix: {f'{C}contacts_{M}polys_D{D}'}: {time_stamp.strftime('%H:%M:%S')}") 
        start_time = time.time()
        
        # Scale/normalize the data 
        arr = utils_math.scale_data(arr, technique='simple')
        num_entries = D
        gen_dict = {'arr': arr, 'num_generations': args.gi_gen, 'population_size': args.gi_pop, 
                              'num_parents': args.gi_par, 'mutation_rate': args.gi_mr, 'num_entries': num_entries, 
                              'adaptive_mutation': False, 'random_children_rate': args.gi_cr, 
                              'direct_mutants_rate': args.gi_dmr, 'print_every': 50}
        
        GGObj = GreedyGenetic(gen_dict, greedy_it=args.greedy_it, num_greedy=args.num_greedy)
        
        logger.info(f"Matrix shape: {arr.shape}")
        logger.info(f"Gen Dict: ")
        logger.info(gen_dict)
        best_results = GGObj.run_genetic_simulation()
        logger.info(f"Best Generation: {best_results['best_generation']}")
        logger.info(f"Best Volume: {best_results['best_score']}")
        logger.info(f"Selected Indices: {best_results['best_indices']}")
        logger.info(f"Total execution time: {time.time() - start_time}")
        logger.info("*"*20)
        logger.handlers.clear()

