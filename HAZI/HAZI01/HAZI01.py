# %%
#Create a function that returns with a subsest of a list.
#The subset's starting and ending indexes should be set as input parameters (the list aswell).
#return type: list
#function name must be: subset
#input parameters: input_list,start_index,end_index

# %%
def subset(input_list, start_index, end_index):
    return input_list[start_index:end_index]

# %%
#Create a function that returns every nth element of a list.
#return type: list
#function name must be: every_nth
#input parameters: input_list,step_size

# %%
def every_nth(input_list, step_size):
    return input_list[::step_size]

# %%
#Create a function that can decide whether a list contains unique values or not
#return type: bool
#function name must be: unique
#input parameters: input_list

# %%
def unique(input_list):
    inspected = []
    for element in input_list:
        if element not in inspected:
            inspected.append(element)
        else:
            return False
    return True

# %%
#Create a function that can flatten a nested list ([[..],[..],..])
#return type: list
#fucntion name must be: flatten
#input parameters: input_list

# %%
def flatten(input_list):
    return [element for elements in input_list for element in elements]

# %%
#Create a function that concatenates n lists
#return type: list
#function name must be: merge_lists
#input parameters: *args


# %%
def merge_lists(*args):
    return [element for elements in args for element in elements]

# %%
#Create a function that can reverse a list of tuples
#example [(1,2),...] => [(2,1),...]
#return type: list
#fucntion name must be: reverse_tuples
#input parameters: input_list

# %%
def reverse_tuples(input_list):
    return [tuples[::-1] for tuples in input_list]

# %%
#Create a function that removes duplicates from a list
#return type: list
#fucntion name must be: remove_tuplicates
#input parameters: input_list

# %%
def remove_duplicates(input_list):
    unique_list = []
    for element in input_list:
        if element not in unique_list:
            unique_list.append(element)
    
    return unique_list

# %%
#Create a function that transposes a nested list (matrix)
#return type: list
#function name must be: transpose
#input parameters: input_list

# %%
def transpose(input_list):
    transposed_matrix = list(zip(*input_list))
    return [list(elements) for elements in transposed_matrix]


# %%
#Create a function that can split a nested list into chunks
#chunk size is given by parameter
#return type: list
#function name must be: split_into_chunks
#input parameters: input_list,chunk_size

# %%
import math
def split_into_chunks(input_list, chunk_size):
    chunked_list = []
    for i in range(0, math.ceil(len(input_list) / chunk_size)):
        chunked_list.append(input_list[i*chunk_size:min([len(input_list), i*chunk_size+chunk_size])])
    return chunked_list

# %%
#Create a function that can merge n dictionaries
#return type: dictionary
#function name must be: merge_dicts
#input parameters: *dict

# %%
def merge_dicts(*dict):
    merged_dict = {}
    for single_dict in dict:
        for key, value in single_dict.items():
            merged_dict[key] = value
    
    return merged_dict

# %%
#Create a function that receives a list of integers and sort them by parity
#and returns with a dictionary like this: {"even":[...],"odd":[...]}
#return type: dict
#function name must be: by_parity
#input parameters: input_list

# %%
def by_parity(input_list):
    return {
        "even": [element for element in input_list if element % 2 == 0],
        "odd": [element for element in input_list if element % 2 == 1]
    }

# %%
#Create a function that receives a dictionary like this: {"some_key":[1,2,3,4],"another_key":[1,2,3,4],....}
#and return a dictionary like this : {"some_key":mean_of_values,"another_key":mean_of_values,....}
#in short calculates the mean of the values key wise
#return type: dict
#function name must be: mean_key_value
#input parameters: input_dict

# %%
def mean_key_value(input_dict):
    mean_dict = {}
    for key, list in input_dict.items():
        mean_dict[key] = sum(list) / len(list)
    return mean_dict

# %%
#If all the functions are created convert this notebook into a .py file and push to your repo


