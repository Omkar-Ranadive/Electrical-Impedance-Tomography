import torch 
import shutil
import matplotlib.pyplot as plt 
import seaborn as sns 

sns.set_theme(style='whitegrid')

def save_checkpoint(state, is_best, EXP_DIR, filename):
    torch.save(state, str(EXP_DIR / filename))
    if is_best:
        shutil.copyfile(str(EXP_DIR / filename), str(EXP_DIR / 'model_best.tar'))
