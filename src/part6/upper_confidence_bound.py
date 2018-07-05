import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

data = pd.read_csv('Ads_CTR_Optimisation.csv')

d = 10
n = data.shape[0]

ads_selected = []
numbers_of_displays = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

for n in range(n):
    ad = 0
    max_upper_bound = 0
    for i in range(d):
        if numbers_of_displays[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_displays[i]
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / numbers_of_displays[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i

    ads_selected.append(ad)
    numbers_of_displays[ad] += 1
    reward = data.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward

print(total_reward)

# visualization
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of time each ad was selected')
plt.show()