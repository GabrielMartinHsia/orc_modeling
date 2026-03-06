#SIMPLE MONTE-CARLO SIMULATION:
#REF: https://youtu.be/slbZ-SLpIgg

'''
You have two tasks to complete in the next 9 hours, expected durations are as follows:

    Task A: 1 to 5 hours
    Task B: 2 to 6 hours

What is the probability you will finish both tasks within the alotted 9 hours?
     
Assumptions:
    Tasks are independent - completion of one has no bearing on completion of the other.
    Task duration ranges follow a uniform distribution, i.e., flat line - there is equal likelihood that each task duration will be anywhere within the stated range.
'''

import numpy as np
import matplotlib.pyplot as plt
sims = 1000000

A = np.random.uniform(1, 5, sims)
B = np.random.uniform(2, 6, sims)

duration = A + B

print((duration > 9).sum()/sims)

plt.figure(figsize=(3, 1.5))
plt.hist(duration, density=True)
plt.axvline(9, color='r')
plt.show()

