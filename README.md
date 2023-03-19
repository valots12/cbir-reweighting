# Implementation of the paper "Feature re-weighting in Content-based image retrieval"

***Part of Digital Signal and Image Management project | UniMiB***

The aim of the project is to provide a human-in-the-loop re-weighting process for a CBIR process. The reference for the the retrieval architecture is the following:
*Das, G., Ray, S., & Wilson, C. (2006). Feature re-weighting in content-based image retrieval. In Image and Video Retrieval: 5th International Conference, CIVR 2006, Tempe, AZ, USA, July 13-15, 2006.*

Main concepts of the paper:
- Use of the previous neural network as feature extractor
- Feature normalization using 3 time std and forcing of each range in the interval [0,1]
- Use of weighted Minkowski distance as similarity measure
- Update of the query results according to user preferences

Also, a simple HTML interface is available for test using the FGVC-Aircraft-100 dataset. A DenseNet201 Neural Network trained on the training set is used as feature extractor. The number of similar images detected at each step is set to 20, while our interface shows only the best 6 ones.

<img src="https://user-images.githubusercontent.com/63108350/226201266-35918085-7344-42bb-b958-5d6ee4ad936c.mp4">

Three different methods are used:

### 1. Rebalancing type 1 

<img src="[Images/akaze_example.jpg](https://user-images.githubusercontent.com/63108350/226203376-fe61aca2-aa52-4964-8773-f025bad4e1a6.png)" width=20%>

New weight for the i-th feature is equal to the division between the standard deviation over the 20 retrieved images and the standard deviation over the relevant images at the previous round.

### 2. Rebalancing type 2

![Screenshot_20230210_071607](https://user-images.githubusercontent.com/63108350/226203388-4fdd1599-18b3-416e-b281-3cbd234c6998.png)
![Screenshot_20230210_071552](https://user-images.githubusercontent.com/63108350/226203391-6c2486f0-ad83-4cf1-8a4f-64f5cb08fe13.png)

New weight for the i-th feature is equal to the division between the sigma quantity defined in the second formula, that depends on the dominant range, and the standard deviation over the relevant images at the previous round.

### 3. Rebalancing type 3

![Screenshot_20230210_130548](https://user-images.githubusercontent.com/63108350/226203398-ad5c9e48-971b-4b7c-84b6-16169a70e15f.png)

New weight for the i-th feature is equal to the the delta value defined in the previous slide by the weights of type 1.

## Results
