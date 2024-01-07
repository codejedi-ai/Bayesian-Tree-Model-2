from factor import Factor

# make a factor that takes in 3 variables
factor = Factor(['Sunny', 'Cold', 'Headache'])
# Sunny	Cold	 Headache	probability
#0	0	0	0.576
#0	0	1	0.008
#0	1	0	0.144
#0	1	1	0.072
#1	0	0	0.108
#1	0	1	0.012
#1	1	0	0.016
#1	1	1	0.108
factor.set_probability(0.576, Sunny=0, Cold=0, Headache=0)
factor.set_probability(0.008, Sunny=0, Cold=0, Headache=1)
factor.set_probability(0.144, Sunny=0, Cold=1, Headache=0)
factor.set_probability(0.072, Sunny=0, Cold=1, Headache=1)
factor.set_probability(0.108, Sunny=1, Cold=0, Headache=0)
factor.set_probability(0.012, Sunny=1, Cold=0, Headache=1)
factor.set_probability(0.016, Sunny=1, Cold=1, Headache=0)
factor.set_probability(0.108, Sunny=1, Cold=1, Headache=1)
# alright now we can use this factor to calculate the probability of any condition
factor.print_table()
# test get probability
print(factor.get_probability(Sunny=1, Cold=1, Headache=1))

# test mul
f = Factor(['a', 'b'])
f.set_probability(0.9, a=1, b=1)
f.set_probability(0.1, a=1, b=0)
f.set_probability(0.4, a=0, b=1)
f.set_probability(0.6, a=0, b=0)
g = Factor(['b', 'c'])
g.set_probability(0.7, b=1, c=1)
g.set_probability(0.3, b=1, c=0)
g.set_probability(0.2, b=0, c=1)
g.set_probability(0.8, b=0, c=0)
print('h(a, b, c):')
h = f * g
h.print_table()
print(h.get_probability(a=1, b=1, c=1))
print('h\'(a, b, c)')
h_prime = g * f
h_prime.print_table()
# test sum out
print('f(b):')
f.sum_out('a').print_table()

# test restrict
print('f(a=1):')
f.restrict('a', 1).print_table()


# test manual variable elimination
# f_1(A):
# a | 0.9
# ~a | 0.1
# f_2(A,B):
# ab | 0.9
# a~b | 0.1
# ~ab | 0.4
# ~a~b | 0.6
# f_3(B,C):
# bc | 0.7
# b~c | 0.3
# ~bc | 0.2
# ~b~c | 0.8
# f_4(B):
# b | 0.85
# ~b | 0.15
# f_5(C):
# c | 0.625
# ~c | 0.375
f_1 = Factor(['A'])
f_1.set_probability(0.9, A=1)
f_1.set_probability(0.1, A=0)
f_2 = Factor(['A', 'B'])
f_2.set_probability(0.9, A=1, B=1)
f_2.set_probability(0.1, A=1, B=0)
f_2.set_probability(0.4, A=0, B=1)
f_2.set_probability(0.6, A=0, B=0)
f_3 = Factor(['B', 'C'])
f_3.set_probability(0.7, B=1, C=1)
f_3.set_probability(0.3, B=1, C=0)
f_3.set_probability(0.2, B=0, C=1)
f_3.set_probability(0.8, B=0, C=0)
f_4 = Factor(['B'])
f_4.set_probability(0.85, B=1)
f_4.set_probability(0.15, B=0)
f_5 = Factor(['C'])
f_5.set_probability(0.625, C=1)
f_5.set_probability(0.375, C=0)
print('f_1:')
f_1.print_table()
print('f_2:')
f_2.print_table()
print('f_3:')
f_3.print_table()
print('f_4:')
f_4.print_table()
print('f_5:')
f_5.print_table()

f_6 = f_1 * f_2
print('f_6:')
f_6.print_table()
f_7 = f_6.sum_out('A')
print('f_7:')
f_7.print_table()
# assert f_7 have same table as f_4
assert f_7 == f_4
from factor import variable_elimination
print('------------------------------------------------------------------variable elimination:')
# variable_elimination([f_1, f_2, f_3], ['A', 'B']).print_table()
variable_elimination([f_1, f_2, f_3], ['B'], {'C':1}).print_table()