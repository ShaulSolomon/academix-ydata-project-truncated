<h3>Meeting with Etam - April 30, 2020</h3>

1. The similarity matrix is currently normalized between 0 and 1, but Etam recommended we try to normalize it at 0 with std 1.
2. **Do: We want a few use cases to see if they work, but for getting metrics, we need to run the example on many subcases**
3. We need to learn the best epsilon. By that once we have the many examples, splitting the data into val and test could help us learn the right epsilon
4. Explore the weights - see how much value is given for the different features
5. **Do: Need to discuss how to deal with missing data - a) Seeing which values are often missing b) creating some rule-based criteria for different weights based off what data is available. (Even 5 mst common combination)
6. **Do: See what happens if we try to learn the weights on disambiguated authors....**
7. Explore playing with features - seeing if they have full names or not...


<h3>Meeting with Etam - April 20, 2020</h3>

1. We explored the data and found that 'Name Disambiguation' happens to 10% of the data, which is enough to acknowledge that its a big enough problem that we can't just randomly assume that uninspected authors are disambiguated.
2. Knowing where for a given `author_last_name`, there is only one `pmid`, we are certain that this is really just one person. We can use them for use cases.
~~3. **Do: Once we get the models working, we need to specify very clear use-cases (ex: taking 30 authors with 10 papers each, taking 3 authors with many papers each)**~~
4. We agreed that adding the 'State' of the Researched instead of the Country (due to the bias of the labeled dataset) would allow us to mimic the country feature perhaps to a better level of complexity (there is more co-relation between country and author than state and author theoretically).

<h3>Meeting with Yuval - April 13, 2020</h3>

1. Even though there are certain benefits to cluster homogeniaty as a whole - it was agreed upon that using the greedy algorithm to assign clusters to researchers is actually a good metric (especially as we add mis-integration and mis-separation to help understand in what ways is the method succesful).
~~2. Besides for a final metric, he suggested we "see how each author is distributed by the clusters' which is basically identical to mis-separation, but will help see exactly what happened in a given case.~~
3. Finally, at-least in the LR model, we can use a single 'loss' - the F1 Score - to allow us to help tune the hyperparameters of the LR and clustering algorithm within a single pipeline.

<h3>Meeting with Etam - April 9, 2020</h3>

1. ~~Do: Look at the labeled dataset and see how many disambiguated authors we have and how many papers they have.~~
2. ~~To discuss with Yuval about the scoring metric, whether to keep the applied metric or do something more generic (Wasserstein Distance)~~
