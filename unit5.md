Unit5: Jaccard Coefficient Calculations

In Unit 5 I was given the following table which is used to calculate the Jaccard coefficients.

| Name  | Gender | Fever | Cough | Test-1 | Test-2 | Test-3 | Test-4 |
|-------|--------|-------|-------|--------|--------|--------|--------|
| Jack  | M      | Y     | N     | P      | N      | N      | A      |
| Mary  | F      | Y     | N     | P      | A      | P      | N      |
| Jim   | M      | Y     | P     | N      | N      | N      | A      |

To calculate them I have written this Python code:

# Define the test results for each person
jack_tests = ['Y', 'N', 'P', 'N', 'N', 'A']
mary_tests = ['Y', 'N', 'P', 'A', 'P', 'N']
jim_tests = ['Y', 'P', 'N', 'N', 'N', 'A']

# Function to calculate Jaccard coefficient
def jaccard_coefficient(set1, set2):
    # Exclude ambiguous results ('A') and calculate intersection and union
    intersection = sum(1 for a, b in zip(set1, set2) if a == b and a != 'A')
    union = sum(1 for a, b in zip(set1, set2) if a != 'A' or b != 'A')
    # Return the Jaccard coefficient
    return intersection / union if union > 0 else 0

# Calculate Jaccard coefficients for each pair
jaccard_jack_mary = jaccard_coefficient(jack_tests, mary_tests)
jaccard_jack_jim = jaccard_coefficient(jack_tests, jim_tests)
jaccard_jim_mary = jaccard_coefficient(jim_tests, mary_tests)

The outcomes:

Jaccard coefficient (Jack, Mary): 0.500
Jaccard coefficient (Jack, Jim): 0.600
Jaccard coefficient (Jim, Mary): 0.167

The Jaccard coefficient provides a numerical value to represent the similarity between two sets of data. The result for the pair (Jack, Jim) is 0.6, which indicates the highest similarity. This suggests their test results are relatively aligned. The pair (Jim, Mary) is 0.167, which indicates the lowest similarity. That implies that their test results differ significantly.

By excluding ambiguous values (e.g., 'A') the Jaccard coefficient can handle uncertain data as it focuses only on relevant and definitive comparisons.

If we consider the power of these calculations there are some implications that could be drawn outside of the math or the context of machine learning alone. When we use the Jaccard coefficient in other realms of society, particularly when applying it to fields such as healthcare, education, criminal justice, marketing, and social sciences we can see a few potential outcomes:

- Improve decision-making: By analysing test results (e.g., for patient outcomes in health care) we can spot similarities between cases more reliably. This can  guide diagnosis and treatment. For instance, if Jack and Jim have a high similarity, their conditions may respond to similar interventions.

- Spotting Inequity: The Jaccard coefficient can spot significant unsimilarities. We can use these calculations to spot inequity which can also be used to find important venues to intervene.

These two points could make decision-making more fair and efficient. However, there are of course also risks involved in relying on this method without considering context and other data. If we would use the Jaccard coefficient to extrapolate profiles of offenders and non-offenders we could see potential issues arise, such as reinforcing stereotypes or systemic biases present in the underlying data. For instance, if the dataset used to calculate similarity is skewed due to historical disparities in law enforcement or judicial processes, the results could unfairly associate certain demographic groups with higher risk profiles. 

The assumptions about the benefits and risks of using the Jaccard coefficient are supported by Niwattanakul et al. (2013), who highlight its efficiency in measuring similarity between datasets while also pointing out its limitations, e.g., in sensitivity to data inconsistencies like redundancy or errors.

References:

Niwattanakul, S., Singthongchai, J., Naenudorn, E. and Wanapu, S., 2013. Using of Jaccard coefficient for keywords similarity. Proceedings of the International MultiConference of Engineers and Computer Scientists (IMECS), 1, pp.380-384. Available at: https://www.iaeng.org/publication/IMECS2013/IMECS2013_pp380-384.pdf [Accessed 25 December 2024].


