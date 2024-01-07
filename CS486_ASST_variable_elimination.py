import numpy as np

'''
Implement the variable elimination algorithm by coding the
following functions in Python. Factors are essentially 
multi-dimensional arrays. Hence use numpy multidimensional 
arrays as your main data structure.  If you are not familiar 
with numpy, go through the following tutorial: 
https://numpy.org/doc/stable/user/quickstart.html
'''



######### restrict function
# Tip: Use slicing operations to implement this function
#
# Inputs: 
# factor -- multidimensional array (one dimension per variable in the domain)
# variable -- integer indicating the variable to be restricted
# value -- integer indicating the value to be assigned to variable
#
# Output:
# resulting_factor -- multidimensional array (the dimension corresponding to variable has been restricted to value)
#########
def restrict(factor,variable,value):
	resulting_factor_shape = list(factor.shape)
	if resulting_factor_shape[variable] == 1:
		return factor
	# dummy result until the function is filled in
	slice_obj = [slice(None)] * factor.ndim
	resulting_factor = np.copy(factor[tuple(slice_obj)])
	slice_obj[variable] = value
	resulting_factor = resulting_factor[tuple(slice_obj)]
	# get the shape of the factor and change the variable dimension to 1
	# get resulting_factor shape into a list

	resulting_factor_shape[variable] = 1
	# reshape the resulting_factor to the resulting_factor_shape
	# turn resulting_factor_shape into a tuple
	resulting_factor_shape = tuple(resulting_factor_shape)

	resulting_factor = np.reshape(resulting_factor,resulting_factor_shape)

	return resulting_factor

######### sumout function
# Tip: Use numpy.sum to implement this function
#
# Inputs: 
# factor -- multidimensional array (one dimension per variable in the domain)
# variable -- integer indicating the variable to be summed out
#
# Output:
# resulting_factor -- multidimensional array (the dimension corresponding to variable has been summed out)
#########
def sumout(factor,variable):

	# dummy result until the function is filled in

	resulting_factor = np.sum(factor, axis=variable)
	resulting_factor_shape = list(factor.shape)
	resulting_factor_shape[variable] = 1
	# reshape the resulting_factor to the resulting_factor_shape
	# turn resulting_factor_shape into a tuple
	resulting_factor_shape = tuple(resulting_factor_shape)

	resulting_factor = np.reshape(resulting_factor, resulting_factor_shape)
	print("\\sum_{" + str(var_map_string[variable]) + "}" + "f_{" + str(variable) + "}")
	return resulting_factor

def print_factor(factor, factor_name = "value"):
	# generate a string f_{factor_name} using string formatting
	# print the string
	# print the factor

	shape_list = list(factor.shape)

	def print_array_table_ignore_one(arr, params=None):
		# Get the shape of the array
		# ret is a string
		ret = ""
		shape = arr.shape

		# Determine the number of dimensions
		num_dims = len(shape)

		# Generate default parameter names if not provided
		if params is None:
			params = [f"param{i + 1}" for i in range(num_dims)]

		# Filter out parameters with only one dimension
		params = [param for param, dim in zip(params, shape) if dim > 1]

		# Calculate the total number of elements
		total_elements = np.prod(shape)

		# Print the table header
		header = " | ".join(params + [factor_name])
		# append header like it is print(header)
		ret += header + "\n"
		# same goes with print("-" * len(header))
		ret += "-" * len(header) + "\n"

		# Print the array values row by row
		for i in range(total_elements):
			# Calculate the indices in each dimension
			indices = np.unravel_index(i, shape)

			# Get the parameter values for the current indices
			param_values = [str(indices[j]) for j in range(num_dims) if shape[j] > 1]

			# Get the corresponding value in the array
			value = str(arr[indices])

			# Print the parameter values and the corresponding array value
			row = " | ".join(param_values + [value])
			# print(row)
			ret += row + "\n"

		return ret
	# get factor parameters as list of strings
	params = [var_map_string[i] for i in range(len(shape_list))]
	print(print_array_table_ignore_one(factor, params))
	# print the factor





######### multiply function
# Tip: take advantage of numpy broadcasting rules to multiply factors with different variables
# See https://numpy.org/doc/stable/user/basics.broadcasting.html
#
# Inputs: 
# factor1 -- multidimensional array (one dimension per variable in the domain)
# factor2 -- multidimensional array (one dimension per variable in the domain)
#
# Output:
# resulting_factor -- multidimensional array (elementwise product of the two factors)
#########
def multiply(factor1,factor2):
	# All the fectors must be in thier right shape and dimension
	# The length of the sampe tuple must be the same throughtout all of the calculations
	# dummy result until the function is filled in
	resulting_factor = np.multiply(factor1, factor2)
	return resulting_factor
# write some tests for factor multiplication
factor1 = np.array([[1, 2, 3], [4, 5, 6]])
factor2 = np.array([[1, 2, 3], [4, 5, 6]])
# write an assert
assert np.all(multiply(factor1, factor2) == np.array([[1, 4, 9], [16, 25, 36]]))
######### normalize function
# Tip: divide by the sum of all entries to normalize the factor
#
# Inputs: 
# factor -- multidimensional array (one dimension per variable in the domain)
#
# Output:
# resulting_factor -- multidimensional array (entries are normalized to sum up to 1)
#########
def normalize(factor):
	sum_of_elements = np.sum(factor)

	# Divide the array by the sum of its elements
	resulting_factor = np.divide(factor, sum_of_elements)
	return resulting_factor
import numpy as np
import pandas as pd

def create_array_from_dataframe(df, params=None):
    # Extract the parameter names from the column names
    if params is None:
        params = list(df.columns[:-1])

    # Initialize an empty dictionary to store the indices
    indices_dict = {param: [] for param in params}

    # Parse the DataFrame rows and extract the parameter values and corresponding array values
    for _, row in df.iterrows():
        param_values = tuple(row[params])
        array_value = row[df.columns[-1]]

        for param, value in zip(params, param_values):
            indices_dict[param].append(value)

    # Create the array from the collected indices
    shape = tuple(len(indices_dict[param]) for param in params)
    arr = np.zeros(shape, dtype=df.dtypes[-1])

    # Assign the array values based on the indices
    indices = [indices_dict[param] for param in params]
    arr[indices] = df[df.columns[-1]].to_numpy()

    return arr

# Example usage
data = {
    "row": [0, 0, 1, 1],
    "column": [0, 1, 0, 1],
    "value": [1, 2, 3, 4]
}

df = pd.DataFrame(data)
arr = create_array_from_dataframe(df)
print(arr)

######### inference function
# Tip: function that computes Pr(query_variables|evidence_list) by variable elimination.  
# This function should restrict the factors in factor_list according to the
# evidence in evidence_list.  Next, it should sumout the hidden variables from the 
# product of the factors in factor_list.  The variables should be summed out in the 
# order given in ordered_list_of_hidden_variables.  Finally, the answer should be
# normalized to obtain a probability distribution that sums up to 1.
#
#Inputs: 
#factor_list -- list of factors (multidimensional arrays) that define the joint distribution of the domain
#query_variables -- list of variables (integers) for which we need to compute the conditional distribution
#ordered_list_of_hidden_variables -- list of variables (integers) that need to be eliminated according to thir order in the list
#evidence_list -- list of assignments where each assignment consists of a variable and a value assigned to it (e.g., [[var1,val1],[var2,val2]])
#a
#Output:
#answer -- multidimensional array (conditional distribution P(query_variables|evidence_list))
#########
def inference(factor_list, query_variables, ordered_list_of_hidden_variables, evidence_list):
	# print("called inference" + str(query_variables) + str(ordered_list_of_hidden_variables) + str(evidence_list))
	# Eliminate hidden variables
	# make list for evidence variables
	# make list for hidden variables
	# make list for query variables
	# make list for ordered variables
	evidence_list_variables = [evidence[0] for evidence in evidence_list]
	for evidence in evidence_list:
		# restrict the factors in factor_list according to the evidence in evidence_list
		# get the variable and value from the evidence
		variable = evidence[0]
		value = evidence[1]
		# restrict the factors in factor_list according to the evidence in evidence_list
		factor_list = [restrict(factor, variable, value) for factor in factor_list]
		# There could be factors that do not have take variable as a parameter
	# need to remove duplicates
	def contains(list,np_array):
		for element in list:
			if np.array_equal(element,np_array):
				return True
		return False

	factor_list_new = []
	for factor in factor_list:
		if not contains(factor_list_new,factor):
			factor_list_new.append(factor)
	factor_list = factor_list_new

	# for factor in factor_list:
		# print_factor(factor, "s")

	for variable in ordered_list_of_hidden_variables:

		if variable in query_variables or variable in evidence_list_variables:
			continue
		# get all the factors in which the variable appears
		factors_to_multiply = []
		# get all the
		# print summing out var to string variable
		# print("summing out on:" + str(var_map_string[variable]))

		# take all the factors in which the variable appears
		# that is when the dimension that is corrusponding to the variable is not 1
		factors_to_multiply = [factor for factor in factor_list if factor.shape[variable] != 1]
		if factors_to_multiply == []:
			continue

		# now make a new factor that multiplies all the factors in factors_to_multiply using the function multiply
		# initialize the new multiply factor to be a numpy array of all ones with the shape (1....1) of factor shape length
		multiply_factor = factors_to_multiply[0]
		# multiply all the factors in factors_to_multiply
		for factor in factors_to_multiply[1:]:
			multiply_factor = multiply(multiply_factor, factor)
		# sumout the variable from the multiply_factor
		multiply_factor = sumout(multiply_factor, variable)
		# normalize the multiply_factor
		multiply_factor = normalize(multiply_factor)
		# remove all the factors in factors_to_multiply from factor_list
		# print_factor(multiply_factor, "multiply_factor")


		factor_list = [factor for factor in factor_list if not contains(factors_to_multiply,factor)]








			# remove_all(factor_list, factor)
		# append the multiply_factor to factor_list
		# if
		if multiply_factor.size != 1:
			factor_list.append(multiply_factor)
		# print the factor_list
	# outside of the for loop
	query_factor = factor_list[0]
	for factor in factor_list[1:]:
		query_factor = multiply(query_factor, factor)
		# right now I can only consider querries of one variable.
		# I need to make it so that it can handle querries of multiple variables
	# normalize the query_factor
	query_factor = normalize(query_factor)
	# print_factor(query_factor, "query_{factor}")
	return query_factor

# #factor_list -- list of factors (multidimensional arrays) that define the joint distribution of the domain zipped as (name, factor)
def inference_factor_list_wName(factor_list, query_variables, ordered_list_of_hidden_variables, evidence_list):
	calculate_string = ""

	factor_count = len(factor_list)
	evidence_list_variables = [evidence[0] for evidence in evidence_list]
	for evidence in evidence_list:
		# restrict the factors in factor_list according to the evidence in evidence_list
		# get the variable and value from the evidence
		variable = evidence[0]
		value = evidence[1]
		# restrict the factors in factor_list according to the evidence in evidence_list
		# factor_list = [restrict(factor, variable, value) for factor in factor_list]
		new_factor_list = []
		for (factor_name, factor) in factor_list:
			new_name = f"f_{factor_count}"
			new_factor = restrict(factor, variable, value)
			new_factor_list.append((new_name, new_factor))
			factor_count += 1
		factor_list = new_factor_list
		# There could be factors that do not have take variable as a parameter
	# need to remove duplicates
	def contains(list,np_array):
		for element in list:
			if np.array_equal(element,np_array):
				return True
		return False

	def factor_params(factor):
		return tuple([var_map_string[variable] for variable in factor.shape if variable != 1])
	for variable in ordered_list_of_hidden_variables:
		# print "\sum_{variable as string}"

		if variable in query_variables or variable in evidence_list_variables:
			continue
		calculate_string += f"\\sum_{{{var_map_string[variable]}}}"
		# get all the factors in which the variable appears
		factors_to_multiply = []
		# get all the
		# print summing out var to string variable
		# print("summing out on:" + str(var_map_string[variable]))

		# take all the factors in which the variable appears
		# that is when the dimension that is corrusponding to the variable is not 1
		factors_to_multiply = []
		for (factor_name, factor) in factor_list:
			if factor.shape[variable] != 1:
				factors_to_multiply.append((factor_name, factor))

		if factors_to_multiply == []:
			continue
		# make a tuple that reutrns the ith element as the name of variable
		# (2,2,2,2,2) -> (Acc,Fraud, Trav,FP,OP,PT)
		# we ignore the elements that are ones



		# now make a new factor that multiplies all the factors in factors_to_multiply using the function multiply
		# initialize the new multiply factor to be a numpy array of all ones with the shape (1....1) of factor shape length
		factors_to_multiply_with_names = factors_to_multiply
		for (factor_name, factor) in factors_to_multiply_with_names:
			calculate_string = calculate_string + f"{factor_name}" + str(factor_params(factor))
		factors_to_multiply = [factor for (name, factor) in factors_to_multiply]
		multiply_factor = factors_to_multiply[0]
		# multiply all the factors in factors_to_multiply
		for factor in factors_to_multiply[1:]:
			multiply_factor = multiply(multiply_factor, factor)
		# sumout the variable from the multiply_factor
		multiply_factor = sumout(multiply_factor, variable)
		# normalize the multiply_factor
		multiply_factor = normalize(multiply_factor)
		# remove all the factors in factors_to_multiply from factor_list
		# print_factor(multiply_factor, "multiply_factor")

		# remove used multiply factors from factor_list
		factor_list = [factor for factor in factor_list if not contains(factors_to_multiply,factor[1])]

		multiply_factor_name = f"f_{factor_count}"
		factor_count += 1








			# remove_all(factor_list, factor)
		# append the multiply_factor to factor_list
		# if
		if multiply_factor.size != 1:
			factor_list.append((multiply_factor_name, multiply_factor))
		# print the factor_list
	# outside of the for loop
	for name, factor in factor_list:
		calculate_string = calculate_string + f"{name}" + str(factor_params(factor))

	name, query_factor = factor_list[0]
	for name, factor in factor_list[1:]:
		# print(factor.shape)
		query_factor = multiply(query_factor, factor)
		# right now I can only consider querries of one variable.
		# I need to make it so that it can handle querries of multiple variables
	# normalize the query_factor
	query_factor = normalize(query_factor)
	# print_factor(query_factor, "query_{factor}")
	print(calculate_string)
	return query_factor

# Example Bayes net from the lecture slides: A -> B -> C

# variables
A=0
B=1
C=2
# generate a mapping from variable integers to variable names
var_map_string = {0:'A',1:'B',2:'C'}
variables = np.array(['A','B','C'])

# values
false=0
true=1
values = np.array(['false','true'])

# factors

# Pr(A)
f1 = np.array([0.1,0.9])
f1 = f1.reshape(2,1,1)
print(f"Pr(A)={np.squeeze(f1)}\n")
print_factor(f1, "f1")
# Pr(B|A) how is this differ from Pr(A,B)?
f2 = np.array([[0.6,0.4],[0.1,0.9]])
f2 = f2.reshape(2,2,1)
print(f"Pr(B|A)={np.squeeze(f2)}\n")
print_factor(f2, "f2")
# Pr(C|B)
f3 = np.array([[0.8,0.2],[0.3,0.7]])
f3 = f3.reshape(1,2,2)
print(f"Pr(C|B)={np.squeeze(f3)}\n")
print_factor(f3, "f3")
# multiply two factors
f4 = multiply(f2,f3)
print(f"multiply(f2,f3)={np.squeeze(f4)}\n")

# sumout a variable
f5 = sumout(f2,A)
print(f"sumout(f2,A)={np.squeeze(f5)}\n")

# restricting a factor
f6 = restrict(f2,A,true)
print(f"restrict(f2,A,true)={np.squeeze(f6)}\n")

# inference P(C)
f7 = inference([f1,f2,f3],[C],[A,B],[])
print(f"P(C)={np.squeeze(f7)}\n")



# This example is for the credit card Fraud
#
#
#
#
#
#
#
#
#
var_map_int = {
    'Acc': 0,
    'Fraud': 1,
    'Trav': 2,
    'FP': 3,
    'OP': 4,
    'PT': 5
}
var_map_string = {
	0: 'Acc',
	1: 'Fraud',
	2: 'Trav',
	3: 'FP',
	4: 'OP',
	5: 'PT'
}
# the dimensions go as follows
# (acc, fraud, trav, fp, op, pt)
# now encode the following
# given a list of strings with the variable names {Acc, Fraud, Trav, FP, OP, PT}
# and retunr a tuple that gives 2 as the dimension if hte string is in the list
# and 1 otherwise
def get_dimensions(variables, var_map=var_map_int):
	dimensions = []
	# iterate through var_map
	for key in var_map:
		# if the key is in variables then append 2
		if key in variables:
			dimensions.append(2)
		# else append 1
		else:
			dimensions.append(1)
	return tuple(dimensions)


# every factor must have 6 dimensions

# P(Trav)
trav_prob = np.array([0.95, 0.05])
f1 = trav_prob.reshape(get_dimensions(['Trav']))


# Pr(Fraud | Trav)
cpt = np.array([[0.996, 0.99], [0.004, 0.01]])
tup = get_dimensions(['Fraud', 'Trav'])
f2 = cpt.reshape(tup)


# Pr(FP | Fraud, Trav)
cpt = np.array([
	[[0.99,0.01],[0.1,0.9]],[[0.9,0.1],[0.1,0.9]]
])
tup = get_dimensions(['FP', 'Fraud', 'Trav'])
f3 = cpt.reshape(tup)

# Pr(PT | Acc)
cpt = np.array([[0.99,0.01],[0.1,0.9]])
tup = get_dimensions(['PT', 'Acc'])
f4 = cpt.reshape(tup)
# Print the conditional probability table





# Pr(OP | Acc, Fraud)
cpt = np.array([
	[[0.9,0.1],[0.7,0.3]],[[0.4,0.6],[0.2,0.8]]
])
tup = get_dimensions(['OP', 'Acc', 'Fraud'])
f5 = cpt.reshape(tup)
# Print the conditional probability table

# Pr(OP | Acc, Fraud)
cpt = np.array([
	0.2,0.8
])
tup = get_dimensions(['Acc'])
f6 = cpt.reshape(tup)
# Print the conditional probability table

# def inference(factor_list, query_variables, ordered_list_of_hidden_variables, evidence_list):
# solve for P(F|FP, ¬OP, PT) using inference apply var_map_ints to the variables
factor_list = [f1, f2]
query_variables = ['Fraud']
query_variables = [var_map_int[x] for x in query_variables]
# Trav, FP, Fraud, OP, Acc, PT
ordered_list_of_hidden_variables = ["Trav", "FP", "OP", "Acc", "PT"]
ordered_list_of_hidden_variables = [var_map_int[x] for x in ordered_list_of_hidden_variables]
evidence_list = []
q1 = inference(factor_list, query_variables, ordered_list_of_hidden_variables, evidence_list)
print(f"P(F)={np.squeeze(q1)}\n")

print("P(Fraud| FP, !OP, PT) = ?")
factor_list = [f1, f2, f3, f4, f5, f6]
evidence_list = [('FP', true), ('OP', false), ('PT', true)]
evidence_list = [(var_map_int[x], y) for (x, y) in evidence_list]
q2 = inference(factor_list, query_variables, ordered_list_of_hidden_variables, evidence_list)
print(f"P(Fraud| FP, !OP, PT)={np.squeeze(q2)}\n")

print("P(Fraud| FP, !OP, PT) = ?")
factor_list = [f1, f2, f3, f4, f5, f6]
evidence_list = [('FP', true), ('OP', false), ('PT', true), ('Trav', true)]
evidence_list = [(var_map_int[x], y) for (x, y) in evidence_list]
q3 = inference(factor_list, query_variables, ordered_list_of_hidden_variables, evidence_list)
print(f"P(Fraud| FP, !OP, PT, Trav)={np.squeeze(q3)}\n")


print("P(Fraud| OP) = ?")
factor_list = [f1, f2, f3, f4, f5, f6]

evidence_list = [('OP', true)]
evidence_list = [(var_map_int[x], y) for (x, y) in evidence_list]
q4 = inference(factor_list, query_variables, ordered_list_of_hidden_variables, evidence_list)
print(f"P(Fraud| OP)={np.squeeze(q4)}\n")

def query(factor_list, query_variables, ordered_list_of_hidden_variables, evidence_list):
	# get the evidence list as a string of variable names like "FP, OP, PT"
	# turn the evidence list into a string
	if isinstance(query_variables[0], int):
		query_variables = [var_map_string[x] for x in query_variables]
	query_string = ",".join(query_variables)
	if isinstance(evidence_list[0], int):
		evidence_list = [(var_map_string[x], y) for (x, y) in evidence_list]

	evidence_string = []
	for eve in evidence_list:
		name = eve[0]
		value = eve[1]
		if value:
			evidence_string.append(f"{name}")
		else:
			evidence_string.append(f"\neg {name}")
	evidence_string = ",".join(evidence_string)
	query_string = f"P({query_string} | {evidence_string})"
	# print(f"{query_string} = ?")
	# do the same for the query variables
	# if query_variables is a list of strings then convert to ints
	# print(query_variables)
	# print(evidence_list)
	if isinstance(query_variables[0], str):
		query_variables = [var_map_int[x] for x in query_variables]
	if isinstance(evidence_list[0][0], str):
		evidence_list = [(var_map_int[x], y) for (x, y) in evidence_list]
	if isinstance(ordered_list_of_hidden_variables[0], str):
		ordered_list_of_hidden_variables = [var_map_int[x] for x in ordered_list_of_hidden_variables]
	# noe get the string P(query_string | evidence_string)
	# print(query_variables)
	# print(evidence_list)
	# print the string with query_string = ?


	# if the evidence_list is a list of strings then convert to ints
	# if factor_list is in the form (factor_name, factor) then use inference_factor_list_wName
	if isinstance(factor_list[0], tuple):
		q4 =  inference_factor_list_wName(factor_list, query_variables, ordered_list_of_hidden_variables, evidence_list)
	else:
		q4 = inference(factor_list, query_variables, ordered_list_of_hidden_variables, evidence_list)
	print(f"{query_string}={np.squeeze(q4)}\n")
	return q4

# make a query for Fraud that assumes OP, Trav, and FP are true customlly define the lists
factor_list = [f1, f2, f3, f4, f5, f6]
factor_list_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
# zip the factor list and the names together
factor_list = list(zip(factor_list_names, factor_list))

query_variables = ['Fraud']
ordered_list_of_hidden_variables = ["Trav", "FP", "OP", "Acc", "PT"]
evidence_list = [('OP', true)]
query(factor_list, query_variables, ordered_list_of_hidden_variables, evidence_list)

evidence_list = [('OP', true), ('Trav', true), ('FP', true), ("Acc", true), ("PT", false)]
query(factor_list, query_variables, ordered_list_of_hidden_variables, evidence_list)

evidence_list = [('OP', true), ('Trav', false), ('FP', false), ("Acc", true), ("PT", false)]
query(factor_list, query_variables, ordered_list_of_hidden_variables, evidence_list)
