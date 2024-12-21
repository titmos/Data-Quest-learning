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

def factorial(n, k): # function factorial with k selection
    product = 1
    for x in range(n, n - k, -1):
        product *= x
    return product
perm_3_52 = factorial(52, 3)
perm_4_20 = factorial(20, 4)
perm_4_27 = factorial(27, 4)
