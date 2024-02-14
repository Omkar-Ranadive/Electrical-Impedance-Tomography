import re


def save_friendly(input_string): 
    # Replace all occurrences of "." and "-" with "_"
    modified_string = re.sub(r'[.-]', '_', str(input_string))
    
    return modified_string