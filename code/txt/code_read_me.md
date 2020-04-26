## Overview of what each file in the code does 

<h3>1_Getting_Labeled_Data.ipynb</h3>

Takes zip files from NIH Database, merges them, does a bit of cleaning, and then saves them.

<h3>1_clean_NIH_dataset.ipynb</h3>

Creates a new column `cleaned_name` that we use in subsequent code to merge Academix database and NIH database, that is taken from `PI_NAMEs` and cleaned to be the same format as Academix names.

<h3>1_merged_datasets_for_labeled_data</h3>

Takes the cleaned NIH dataset and merges it with Academix's database based off the same `pmid` and  `cleaned_name`. Sorts the solutions as `enriched_labeled_dataset` and `not_enriched_labeled_dataset`