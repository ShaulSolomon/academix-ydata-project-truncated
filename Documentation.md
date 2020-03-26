<h1>Industry Project Documentation:</h1>

<h3>Objective</h3>

Develop a system to disambiguate biomedical research papers authored by researchers with similar name.

<h3>Methodology</h3>

Train autoencoder network to encode research papers and use it’s latent space as input parameters for unsupervised clustering with DBSCAN

<h3>Roadmap</h3>

- [ ] 1) Find database to be used as benchmark<br>
- [ ] 2) Decide on metric for cluster evaluation<br>
- [ ] 3) Run the current DBSCAN algorithm and obtain the benchmark<br>
- [ ] 4) Setup basic (variational) auto-encoder and run it on mnist/image data<br>
- [ ] 5) Research a VAE with architecture that was designed to solve a similar problem to ours.<br>
- [ ] 6) Find a way to embed the co-authors feature.<br>
- [ ] 7) Train the auto-encoder using the supervised dataset and run a basic DBSCAN to see functionality.<br>
- [ ] 8) Iteratively embed other meta-data features, and possibly do feature engineering (ex: popularity of authors name), comparing to benchmark.<br>
- [ ] 9) Try to use NLP/BERT on the `abstract` and `title` to add to our input features to see if it gives us a better result.<br>
- [ ] 10)Discuss ways to include human annotation (either backend or part of UI)<br>
<br>
<a href="https://drive.google.com/open?id=111D3DuSWclGpsgyv009pgCFsnXrZOJr4">Short Presentation Video</a>
<br>

<h2>Detailed Approach</h2>

<h3> 0) Our own Data </h3>

Our data is originally scraped from the PubMed Dataset, with the <a href="https://www.nlm.nih.gov/bsd/mms/medlineelements.html"> following metadata </a>:


Abstract	(AB)|Copyright Information	(CI) | Affiliation	(AD)|Investigator Affiliation	(IRAD) |Article Identifier	(AID)
------------- | ------------- | ------------- | ------------- | -------------
Author	(AU) | Author Identifier	(AUID) | Full Author	(FAU) | Book Title	(BTI) | Collection Title	(CTI)
Comments/Corrections | Conflict of Interest Statement	(COIS) | Corporate Author	(CN) | Create Date	(CRDT) | Date Completed	(DCOM)
Date Created	(DA) | Date Last Revised	(LR) | Date of Electronic Publication	(DEP) | Date of Publication	(DP) |Edition	(EN)
Editor and Full Editor Name	(ED) (FED) | Entrez Date	(EDAT) | Gene Symbol	(GS) | General Note	(GN) | Grant Number	(GR)
Investigator Name and Full Investigator Name	(IR) (FIR) | ISBN	(ISBN) | ISSN	(IS) | Issue	(IP) | Journal Title Abbreviation	(TA)
Journal Title	(JT) | Language	(LA) | Location Identifier	(LID) | Manuscript Identifier	(MID) | MeSH Date	(MHDA)
MeSH Terms	(MH) | NLM Unique ID	(JID) | Number of References	(RF) | Other Abstract	(OAB) | Other Copyright Information	(OCI)
Other ID	(OID) | Other Term	(OT) | Other Term Owner	(OTO) | Owner	(OWN) | Pagination	(PG)
Personal Name as Subject	(PS) | Full Personal Name as Subject	(FPS) | Place of Publication	(PL) | Publication History Status	(PHST) | Publication Status	(PST)
Publication Type	(PT) | Publishing Model	(PUBM) | PubMed Central Identifier	(PMC) | PubMed Central Release	(PMCR) | PubMed Unique Identifier	(PMID)
Registry Number/EC Number	(RN) | Substance Name	(NM) | Secondary Source ID	(SI) | Source	(SO) | Space Flight Mission	(SFM)
Status	(STAT) | Subset	(SB) | Title	(TI) | Transliterated Title	(TT) | Volume	(VI)
Volume Title	(VTI)

Two import points to mention here: <br>
1) Depending on the field of Academic Research, different protocols are established. In Bio-medical research, practice is to put the LI (Lead Investigator)
as the last name within the set of Authors. **Until 2014, Pubmed metadata only included data on the first author.**
2) Being that there is no strict enforcement of metadata standards, a lot of the metadata is inconsistent (Some papers include middle names some just initials), and a lot of the metadata is missing.

However, due to extensive prior work done by Academix, they have an enriched dataset:


- **INSERT INFORMATION ABOUT ENRICHED DATASET**

Side note: There is still room for improvement to the dataset. Ex: Truncating emails (when available) to the name before @ sign or assigning weights to the MESH Data

<h3> 1) Find database to be used as benchmark </h3>

There are two possible methods we can take to acquiring a large enough dataset to use to test out dataset (~50K pubmed articles).

1) Given that in most cases, two papers written by the same author is actually the same person (besides for a few, which are easily avoidable (having a disproportionate number of papers assigned to them),
we can just take our own dataset, remove the names, and see if our clustering algorithm separates the papers by the right authors. The accuracy isn't perfect, but it would be a decent and usable approximator.

2) Many other attempts have been made to create a reliable dataset. We can explore Google Scholar / Author-ity / NIH PI (Principal Investigator) ID. They dont have an accesible dataset of id's
large enough to resolve our core problem, but they are a viable option for testing/scoring.

**Google Scholar** : The dataset seems to have a bias that the profiles are ensured by the academic themselves.<br>
**Author-ity** : While they offer a search engine, their database is not open to the public.<br>
- **NIH PI ID** : There is no one-clear database with both PI ID's and PMID (Pubmed Article ID), there is a possibility to join the two using shared cells (Article Title).


