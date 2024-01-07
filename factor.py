import numpy as np
# A factor is a function that takes in conditions x_1, x_2, ..., x_k and returns a value,
# the value is the probability of the conditions being true.
class Factor:
    def __init__(self, variables):
        # there are three types of indicies kept here
        # for each variable it would be mapped to an index onto the table
        # for each tuple the value is obtained from the index
        # for tuple iteration variable -> index -> value
        self.variables = variables
        # make a table of 2^k size where k is the number of variables, there would be k dimensions
        # and each dimension would have 2 values
        self.table = np.zeros([2] * len(variables))
        # make a map that maps each variable to its index in the table
        self.var_to_index = {var: i for i, var in enumerate(variables)}
    # given a tuple I want to return a dictionary that maps each variable to its value
    # the ith index of the tuple is the value of the ith variable
    # get conditions obtains from variable to truth value from a given tuple
    # tuple: index -> truth value
    # var_to_index: -> variable -> index
    def _get_condition(self, tuple):
        # assert the length of the tuple is the same as the number of variables
        assert len(tuple) == len(self.variables)
        # get the condition of the factor
        # (**{var: index[var_to_index[var]] for var in new_factor.variables})
        return {var: tuple[self.var_to_index[var]] for var in self.variables}
    def set_probability(self, value, **kwargs):
        # assert the size of kwargs is the same as the number of variables
        assert len(kwargs) == len(self.variables)
        # set the probability of the value to the probability of the conditions being true
        self.table[tuple(kwargs[var] for var in self.variables)] = value
    def print_table(self):
        #print the table in the format of given all the variables # print the probability
        # var_1, var_2, ..., var_k, probability
        # 0, 0, ..., 0, probability
        # 0, 0, ..., 1, probability
        # ...
        # 1, 1, ..., 1,
        print('|'+'|'.join(self.variables + ['probability'])+'|')
        print('|'+'|'.join('-' for i in range(len(self.variables) + 1))+'|')
        for index, value in np.ndenumerate(self.table):
            print('|'+'|'.join(str(i) for i in index) + '|' + str(value)+'|' )
    # get_probability function takes in a dictionary that maps each variable to its value
    # and returns the probability of the conditions being true
    def get_probability(self, **kwargs):
        # print('kwargs', kwargs)
        # get the probability of the conditions being true
        return self.table[tuple(kwargs[var] for var in self.variables)]
    # alright now I need to implement factor multiplication
    # now I need to implement factor multiplication
    # The product of two factors
    # example:
    # f(a, b):
    # ab = 0.9
    # a~b = 0.1
    # ~ab = 0.4
    # ~a~b = 0.6
    # g(b, c):
    # bc = 0.7
    # b~c = 0.3
    # ~bc = 0.2
    # ~b~c = 0.8
    # h = f * g
    # h(a, b, c):
    # abc = 0.63
    # ab~c = 0.27
    # a~bc = 0.02
    # a~b~c = 0.08
    # ~abc = 0.28
    # ~ab~c = 0.12
    # ~a~bc = 0.12
    # ~a~b~c = 0.48
    # alright now I need to implement factor multiplication
    def __mul__(self, other):
        # make a new factor with the union of the variables of the two factors
        new_factor = Factor(list(set(self.variables + other.variables)))
        # make a map that maps each variable to its index in the new factor
        # self.var_to_index = {var: i for i, var in enumerate(variables)}
        # var_to_index = new_factor.var_to_index
        # iterate through the table of the new factor
        for index, value in np.ndenumerate(new_factor.table):
            # index is a tuple of the indices of the new factor
            # which returns truth value based from index
            # thus index -> truth value

            # get the conditions of the new factor
            # conditions = {var: index[var_to_index[var]] for var in new_factor.variables}
            conditions = new_factor._get_condition(index)
            #print('conditions', conditions)
            # get the conditions of the first factor
            conditions_1 = {var: conditions[var] for var in self.variables}
            #print('conditions_1', conditions_1)
            # get the conditions of the second factor
            conditions_2 = {var: conditions[var] for var in other.variables}
            #print('conditions_2', conditions_2)
            # set the probability of the new factor to the product of the probabilities of the two factors
            new_factor.set_probability(self.get_probability(**conditions_1) * other.get_probability(**conditions_2), **conditions)
        # return the new factor
        return new_factor
    # define equality
    # check for the same variables and the same table values small error is allowed
    def __eq__(self, other):
        return self.variables == other.variables and (np.array_equal(self.table, other.table) or np.allclose(self.table, other.table))
    # rewrite the print table function however this time it returns a string
    # make the to string function which involves printing the table
    def __str__(self):
        # make a string
        string = ''
        # add the variables to the string
        string += '|'+'|'.join(self.variables + ['probability'])+'|\n'
        # add the line
        string += '|'+'|'.join('-' for i in range(len(self.variables) + 1))+'|\n'
        # iterate through the table
        for index, value in np.ndenumerate(self.table):
            # add the index
            string += '|'+'|'.join(str(i) for i in index) + '|' + str(value)+'|\n'
        # return the string
        return string
    # sum out a variable, choose a variable to sum up, and the variable must be in the factor
    # example:
    # f(a,b)
    # ab = 0.9
    # a~b = 0.1
    # ~ab = 0.4
    # ~a~b = 0.6
    # f.sum_out('a')
    # f(b)
    # b = 1.3
    # ~b = 0.7
    # alright now I need to implement factor sum out
    def sum_out(self, variable):
        # assert variable is in the factor
        assert variable in self.variables
        # assert variable is a single variable or a string
        assert type(variable) == str

        # make a new factor with the variables of the old factor minus the variable to sum out
        new_factor = Factor([var for var in self.variables if var != variable])
        # make a map that maps each variable to its index in the new factor
        var_to_index = new_factor.var_to_index
        # iterate through the table of the new factor
        for index, value in np.ndenumerate(new_factor.table):
            # get the conditions of the new factor
            conditions = new_factor._get_condition(index)
            # print('conditions', conditions)
            # get the conditions of the old factor
            #print('conditions_old', conditions_old)
            # set the probability of the new factor to the sum of the probabilities of the old factor
            # with the variable to sum out being true and false
            new_factor.set_probability(self.get_probability(**conditions, **{variable: 1}) + self.get_probability(**conditions, **{variable: 0}), **conditions)
        # return the new factor
        return new_factor
    # restrict a variable, choose a variable to restrict, and the variable must be in the factor
    # example:
    # f(a,b)
    # ab = 0.9
    # a~b = 0.1
    # ~ab = 0.4
    # ~a~b = 0.6
    # f.restrict('a', 1)
    # f(b)
    # b = 0.9
    # ~b = 0.4
    # alright now I need to implement factor restrict
    def restrict(self, variable, set_value):
        assert variable in self.variables
        assert type(variable) == str
        assert set_value in [0, 1]
        new_factor = Factor([var for var in self.variables if var != variable])
        for index, value in np.ndenumerate(new_factor.table):
            conditions = new_factor._get_condition(index)
            new_factor.set_probability(self.get_probability(**conditions, **{variable: set_value}), **conditions)
        return new_factor
    # normalize, does not return inplace, it normalizes self
    def normalize(self):
        # get the sum of the probabilities of the factor
        sum = np.sum(self.table)
        # divide each value in the table by the sum
        self.table /= sum
        # return self
        return self

def variable_elimination(Factors, Variables, Evidence = {}):
    # evidence is a dictionary that maps each variable to its value
    # for every factor that contains a variable in evidence restrict the variable to value of evidence
    for factor in Factors:
        # for each variable in the keys of evidence
        for var in Evidence.keys():
            if var in factor.variables:
                factor = factor.restrict(var, Evidence[var])

    for var in Variables:
        # find factors that contain the variable
        factors = [factor for factor in Factors if var in factor.variables]
        # multiply all factors that contain the variable
        new_factor = factors[0]
        for factor in factors[1:]:
            new_factor *= factor
        # sum out the variable
        new_factor = new_factor.sum_out(var)
        # remove the old factors
        for factor in factors:
            Factors.remove(factor)
        new_factor.normalize()
        # add the new factor
        Factors.append(new_factor)

    # assert Factors being length 1
    # the remaining factors refer to only the query variable Q
    # take the product of the remaining factors and normalize
    ret_factor = Factors[0]
    # pop the top factor
    Factors.pop()
    # multiply all the factors
    while len(Factors) > 0:
        ret_factor *= Factors[0]
        Factors.pop()
    # remove all the factors
    assert len(Factors) == 0
    # add the new factor
    Factors.append(ret_factor)
    assert len(Factors) == 1
    ret_factor.normalize()
    return ret_factor
# alright test the thing
