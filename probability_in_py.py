#newest!
#Estimating Probabilities
#Repeating an Experiment

# INITIAL CODE
from numpy.random import seed, randint

seed(1) #to set a random seed for reproducibility 
#Because randint() generates numbers randomly, there's a 50% chance to get a 0 and 50% chance to get a 1 — this helps us introduce in our code a logic for P(H) = 50% and P(T) = 50%
def coin_toss():
    if randint(0,2) == 1: #to generate randomly either a 0 or a 1,2 not inclusive
        return 'HEAD'
    else:
        return 'TAIL'
#If randint(0, 2) returns 1, our coin_toss() function returns 'HEAD', otherwise it returns 'TAIL'
probabilities = []
heads = 0

for n in range(1, 10001):
    outcome = coin_toss()
    if outcome == 'HEAD':
        heads += 1
    current_probability = heads/n
    probabilities.append(current_probability)
print(probabilities[:10])
print(probabilities[-10:])


#Probability Rules
#Sample Space

coin_toss_omega = ['HH', 'HT', 'TH', 'TT' ] #sample space of tossing a normal coin two times. 


#Permutations without replacement
def factorial(n): # function factorial
    product = 1
    for x in range(1, n + 1):
        product *= x
    return product
#test
permutations_1 = factorial(6)
permutations_2 = factorial(52)


#More About Permutations
#Note that we can't use the Permutations = n! formula to calculate the number of permutations for a 5-card poker hand because in our case n = 52, and that'd lead us to a wrong result:
#we have a group of n objects, but we're taking only k objects
# formula below to calculate permutations when we're sampling without replacement and taking only k objects from a group of n objects:
def factorial(n, k): # function factorial with k selection
    product = 1
    for x in range(n, n - k, -1):
        product *= x
    return product
perm_3_52 = factorial(52, 3)
perm_4_20 = factorial(20, 4)
perm_4_27 = factorial(27, 4)

def factorial(n, k): # function factorial with k selection
    product = 1
    for x in range(n, n - k, -1):
        product *= x
    return product

#A password manager software generates 16-character passwords from a list of 127 characters (the list contains numbers, letters, or other symbols)
n_outcomes = factorial(127, 16)
#Assume the sampling from the list is done randomly and without replacement, and find the probability of cracking a password generated by this software if we're using the password "@*AmgJ(UL3Yl726x", which has 16 characters
p_crack_pass = 1/n_outcomes


#Unique Arrangements

def factorial(n):
    final_product = 1
    for i in range(n, 0, -1):
        final_product *= i
    return final_product

def permutation(n, k):
    numerator = factorial(n)
    denominator = factorial(n-k)
    return numerator/denominator
c = permutation(52, 5)/factorial(5)
p_aces_7 = 1/c
c_lottery = permutation(49, 6)/factorial(6)
p_big_prize = 1/c_lottery


#combination
def factorial(n):
    final_product = 1
    for i in range(n, 0, -1):
        final_product *= i
    return final_product
def combinations(n, k):#outputs the number of combinations when we're taking only k objects from a group of n object
    numerator = factorial(n)
    denominator = factorial(k) * factorial(n - k)
    return numerator / denominator
c_18 = combinations(34, 18)
p_Y = 1/c_18
p_non_Y = 1 - p_Y



#Conditional Probability: Intermediate
#The Multiplication Rule

p_ram = 0.0822
p_gl = 0.0184
p_ram_given_gl = 0.0022
p_gl_and_ram = p_gl * p_ram_given_gl
p_non_ram_given_gl = 1 - p_ram_given_gl
p_gl_and_non_ram = p_gl * p_non_ram_given_gl
p_gl_or_ram = p_gl + p_ram - p_gl_and_ram
