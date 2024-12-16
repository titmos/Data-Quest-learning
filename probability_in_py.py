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
