# Implementation of the paper "Feature re-weighting in content-based image retrieval"

***Part of the Digital Signal and Image Management project | UniMiB***

The aim of the project is to provide a human-in-the-loop re-weighting process for a CBIR process. The reference for the retrieval architecture is the following:
*Das, G., Ray, S., & Wilson, C. (2006). Feature re-weighting in content-based image retrieval. In Image and Video Retrieval: 5th International Conference, CIVR 2006, Tempe, AZ, USA, July 13-15, 2006.*

Main concepts of the paper:
- use of the previous neural network as feature extractor;
- feature normalization using 3 time standard deviation and forcing each range in the interval [0,1];
- use of weighted Minkowski distance as similarity measure;
- update of the query results according to user preferences.

Also, a simple HTML interface is available for testing using the FGVC-Aircraft-100 dataset. A DenseNet201 neural network trained on the training set is used as feature extractor. The number of similar images detected at each step is set to 20, while our interface shows only the best six.

<img src="https://user-images.githubusercontent.com/63108350/226201266-35918085-7344-42bb-b958-5d6ee4ad936c.mp4">

Three different methods are used:

### 1. Rebalancing type 1

<img src="https://user-images.githubusercontent.com/63108350/226203376-fe61aca2-aa52-4964-8773-f025bad4e1a6.png" width=50%>

The new weight for the i-th feature is equal to the division between the standard deviation over the 20 retrieved images and the standard deviation over the relevant images at the previous round.

### 2. Rebalancing type 2

<img src="https://user-images.githubusercontent.com/63108350/226203388-4fdd1599-18b3-416e-b281-3cbd234c6998.png" width=50%>
<img src="https://user-images.githubusercontent.com/63108350/226203391-6c2486f0-ad83-4cf1-8a4f-64f5cb08fe13.png" width=25%>

The new weight for the i-th feature is equal to the division between the sigma quantity defined in the second formula, which depends on the dominant range, and the standard deviation over the relevant images at the previous round.

### 3. Rebalancing type 3

<img src="https://user-images.githubusercontent.com/63108350/226203398-ad5c9e48-971b-4b7c-84b6-16169a70e15f.png" width=50%>

The new weight for the i-th feature is equal to the delta value defined in the previous slide by the weights of type 1.

## Results

The table below contains the performance of the methods described according to the top-20 precision metric. To automatate the process, the update of the weights is done considering the labels of each image to simulate the intervention of a human.

| Iteration    | Type 1    | Type 2    | Type 3    |
| ------------ | --------- | --------- | --------- |
| Round 0      | 77.56     | ***77.56*** | ***77.56*** |
| Round 1      | 83.94     | 61.70     | 60.33     |
| Round 2      | 84.56     | 58.84     | 57.35     |
| Round 3      | 85.10     | 59.91     | 57.94     |
| Round 4      | 85.41     | 60.09     | 57.85     |
| Round 5      | ***85.54*** | 60.53     | 57.77     |

<img src="https://user-images.githubusercontent.com/63108350/226281890-6bf182d4-24a7-4ac2-bf3c-9ad91f769bc1.png">

Only the first method leads to an increase in the metric used. For this reason, it is the one used for the interface demo.
