<h3>Meeting with Etam - April 20, 2020</h3>

1. We explored the data and found that 'Name Disambiguation' happens to 10% of the data, which is enough to acknowledge that its a big enough problem that we can't just randomly assume that uninspected authors are disambiguated.
2. Knowing where for a given `author_last_name`, there is only one `pmid`, we are certain that this is really just one person. We can use them for use cases.
3. **Do: Once we get the models working, we need to specify very clear use-cases (ex: taking 30 authors with 10 papers each, taking 3 authors with many papers each)**

<h3>Meeting with Yuval - April 13, 2020</h3>

1. Even though there are certain benefits to cluster homogeniaty as a whole - it was agreed upon that using the greedy algorithm to assign clusters to researchers is actually a good metric (especially as we add mis-integration and mis-separation to help understand in what ways is the method succesful).
2. Besides for a final metric, he suggested we "see how each author is distributed by the clusters' which is basically identical to mis-separation, but will help see exactly what happened in a given case.
3. Finally, at-least in the LR model, we can use a single 'loss' - the F1 Score - to allow us to help tune the hyperparameters of the LR and clustering algorithm within a single pipeline.

<h3>Meeting with Etam - April 9, 2020</h3>

1. ~~Do: Look at the labeled dataset and see how many disambiguated authors we have and how many papers they have.~~
2. ~~To discuss with Yuval about the scoring metric, whether to keep the applied metric or do something more generic (Wasserstein Distance)~~
