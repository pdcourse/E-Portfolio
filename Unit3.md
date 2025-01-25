Unit 3: Correlation and Regression

For this unit, I looked at adding noise to see its effect on correlation specifically. In the provided example file "Unit03 Ex1 covariance_pearson_correlation.ipynb" I have changed the noise in the data2 variable by adding a higher random generation.

data2 = data1 + (5 * randn(1000) + 20)  # Less noise
data2 = data1 + (15 * randn(1000) + 70)  # More noise

For one run, the outcomes for "less noise" and "more noise" are these respectively:


By changing the noise, we can see multiple effects in specific areas:

- Covariance: By adding more noise data2 increases its variability, which can dilute the shared variance between data1 and data2. This leads to a lower covariance value.

- Pearson's Correlation Coefficient: When noise is added to data2, the linear relationship weakens. This reduces the correlation coefficient. In extreme cases of noise, correlation could even approach zero if data2 becomes dominated by random fluctuations.

- Plotting: In the scatter plot, adding noise causes the points to spread out further which causes the relationmship between data1 and data2 to appear less structured.

These outcomes show that real-world data is subject to changes when one variable is influenced by random (or at least hard to control) factors such as measurement errors, environmental influence, etc. This noise can obscure true relationships which can make it more difficult to detect trends. High noise can hide observable correlation this way.

I did further reading on this topic and saw that my assumptions were also mirrored in the literature. Schober et al. (2019) explain that all measurements inherently include a random error component (commonly referred to as noise or uncertainty). Unlike systematic errors, which are consistent and can often be corrected, random errors are unpredictable and vary with each measurement.


References:

Schober et al. (2019): Schober, P., Boer, C. and Schwarte, L.A., 2019. Correlation coefficients: appropriate use and interpretation. Anesthesia & Analgesia, 126(5), pp.1763-1768. Available at: https://www.nature.com/articles/s41598-019-57247-4 [Accessed 1 January 2025].