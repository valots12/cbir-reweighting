# Implementation of the paper "Feature re-weighting in Content-based image retrieval"

***Part of Digital Signal and Image Management project | UniMiB***

The aim of the project is to provide a human-in-the-loop re-weighting process for a CBIR process. The reference for the the retrieval architecture is the following: ***Das, G., Ray, S., & Wilson, C. (2006). Feature re-weighting in content-based image retrieval. In Image and Video Retrieval: 5th International Conference, CIVR 2006, Tempe, AZ, USA, July 13-15, 2006. Proceedings 5 (pp. 193-200). Springer Berlin Heidelberg.***
Also, a simple HTML interface is available for test using the FGVC-Aircraft-100 dataset.

Main concepts of the paper:
- Use of the previous neural network as feature extractor
- Feature normalization using 3 time std and forcing of each range in the interval [0,1]
- Use of weighted Minkowski distance as similarity measure
- Update of the query results according to user preferences

In the following example, 
