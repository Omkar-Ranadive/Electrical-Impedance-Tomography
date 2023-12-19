from genetic_algo import Genetic, EnsembleGenetic
from constants import DATA_PATH, EXP_PATH
import time 
import utils_math
import scipy.io as sio
import logging 
import os 
from datetime import datetime
import argparse 


data_list = [
(4, 6, 2),
(5, 5, 5),
(6, 14, 9),
(7, 14, 14),
(8, 28, 20),
(9, 27, 27),
(11, 44, 44),
(13, 65, 65),
(16, 120, 104),
(6, 6, 2),
(8, 5, 5),
(9, 15, 9),
(11, 14, 14),
(13, 28, 20),
(15, 27, 27),
(18, 44, 44),
(22, 65, 65),
(27, 120, 104),
(7, 6, 2),
(9, 5, 5),
(12, 14, 9),
(14, 14, 14),
(16, 28, 20),
(18, 27, 27),
(23, 44, 44),
(28, 65, 65),
(34, 120, 104),
(8, 6, 2),
(12, 5, 5),
(15, 14, 9),
(19, 14, 14),
(22, 28, 20),
(25, 27, 27),
(32, 44, 44),
(38, 65, 65),
(48, 120, 104)
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=0, type=int, help="Starting index of data list")
    parser.add_argument("--end", default=len(data_list), type=int, help="Ending index of data list")
    parser.add_argument("--genit", default=20, type=int, help="Amount of instance of genetic algorithm")
    args = parser.parse_args()

    # Load the data 
    for mat_info in data_list[args.start:args.end]: 
        C, M, D = mat_info
        print(f"Running for: {C}contacts_{M}polys_D{D}")
        data = sio.loadmat(DATA_PATH / 'OptimizationMatrices' / f'{C}contacts_{M}polys.mat')
        arr = data['JGQ']
        EXP_DIR = EXP_PATH / 'OptimizationMatrices' / f'{C}contacts_{M}polys_D{D}'
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
        gen_dict_instances = {'arr': arr, 'num_generations': 100, 'population_size': 2500, 'num_parents': 200, 
                    'mutation_rate': 0.2, 'num_entries': num_entries, 'adaptive_mutation': False, 
                    'random_children_rate': 0.2, 'direct_mutants_rate': 0.05}
        gen_dict_ensemble = {'arr': arr, 'num_generations': 100, 'population_size': 2000, 'num_parents': 300, 
                    'mutation_rate': 0.4, 'num_entries': num_entries, 'adaptive_mutation': False, 
                    'random_children_rate': 0.3, 'direct_mutants_rate': 0.1}
        EnGenObj = EnsembleGenetic(gen_instances=args.genit, gen_dict_instances=gen_dict_instances, 
                                gen_dict_ensemble=gen_dict_ensemble, logger=logger)
        
        logger.info(f"Matrix shape: {arr.shape}")
        logger.info(f"Gen Dict Instances: ")
        logger.info(gen_dict_instances)
        logger.info(f"Gen Dict Ensemble: ")
        logger.info(gen_dict_ensemble)
        best_results = EnGenObj.run_ensemble_genetics()
        logger.info(f"Best Generation: {best_results['best_generation']}")
        logger.info(f"Best Volume: {best_results['best_score']}")
        logger.info(f"Selected Indices: {best_results['best_indices']}")
        logger.info(f"Total execution time: {time.time() - start_time}")
        logger.info("*"*20)
        logger.handlers.clear()

