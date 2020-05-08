# CS 5293 Spring 2020 Project 3
## Cuisine/Recipe Finder
### By Abilash Ramesh
-------
## Contents:
* Build dataframe / data cleansing
* Feature extraction from dataset
* Data classification
* Display results
----
## Build dataframe / data cleansing:

We load the data provided to us using the yummly.json file. I used to pandas library to load the json file into dataframe for ease of data storage and editing. The data is loaded using the following function:
```python
get_data(filename)
```
The input to the function is the file location. Once the file is loaded, we look into the data structure and see that the column ingredients contains list of ingredients. 

To make sure that the format is readable for the vectorizer, we convert the lists into strings.
``` python
for s in df['ingredients']:
        s = ' '.join(s)
        new.append(s)
    df['ing'] = new
````
Once the string conversion is done, we proceed to the next step where we request the user to provide the ingredients into the given function.
``` python
get_user_ing(df, 'string_of_ingredients')
```
The input has to be a set of strings separated by commas such as the following example:
> 'ing_1, ing_2, ing_3'
Once the dataframe and string is passed into the function, the string is parsed into the dataframe and the dataframe is all set for extracting the feature matrix using the TF-IDF vectorizer.

## Feature extraction from dataset:

We will begin the process of feature extraction by passing the dataframe and the number of clusters into the function as follows:
``` python
train_data(df, n)
```
With the above function, what we do is that we extract the features using the TF-IDF nlp extraction model. The model is specifically chosen as it does a better job at analysing the importance of each ingredient in the distinct cuisines. Once the vectorization process is performed, we obtain the dense matrix. The dense matrix is the split into test and train dataset. We will be using the **K-Nearest Neighbors** classification algorithm for the classification part. 

The dense matrix is split into train and test. The test contains the matirx related to only the new additional ingredients. The train matirx contains all the other recipes provided by the dataset. We use this data to train the model and predict the cuisine for the given ingredients. 

## Data classification:

The data classification is performed using KNN. We have used KNN for our model because:

1. KNN is relatively simple to implement
2. Process of training is much simple
3. Easier to add more data

With the computation complexity in mind, I performed the KNN classification on the train data. The cuisines for the recipes were used as labels for the train data. Once the data was trained, we obtained the model which can be used for prediction purpose. 

For my example, I have taken 8 clusters and thus it will display 8 recipes. The reason for taking 8 clusters is that I felt that it classifies the cuisines much clearly. Given below is an example for the difference between taking 5 clusters and 8 clusters.

With 5 clusters:
``` python
test, mod = train_data(df, 5)
get_results(test, mod)
```
```output
The model predicts that the ingredients resembles french (40.000000 resemblence)

Recipe No: 38052 (0.995421 probable match)
Recipe No: 33387 (1.014186 probable match)
Recipe No: 16446 (1.042872 probable match)
Recipe No: 21589 (1.074481 probable match)
Recipe No: 24719 (1.077459 probable match)

```

From the above result, we can clearly say that thyme and basil are more prominent in Italian cuisine rather than French cuisine
``` python
test, mod = train_data(df, 8)
get_results(test, mod)
```
```output
The model predicts that the ingredients resembles italian (62.500000 resemblence)

Recipe No: 38052 (0.995421 probable match)
Recipe No: 33387 (1.014186 probable match)
Recipe No: 16446 (1.042872 probable match)
Recipe No: 21589 (1.074481 probable match)
Recipe No: 24719 (1.077459 probable match)
Recipe No: 39031 (1.080288 probable match)
Recipe No: 6651 (1.084373 probable match)
Recipe No: 8241 (1.084596 probable match)
```

## Display results:

Once the classification has been performed, the test data is provided and the prediction is performed. I have chosen to take **8 clusters**. The number of clusters can be varied and thus the number of recipes that are obtained is also varied along with it. The input to display the results are as follows:
````python
get_results(test, model)
````
With the above function, we obtain the results according to the number of clusters. 
```output
The model predicts that the ingredients resembles greek (62.500000 resemblence)

Recipe No: 1054 (0.836774 probable match)
Recipe No: 17211 (0.995302 probable match)
Recipe No: 21868 (0.999627 probable match)
Recipe No: 26480 (1.021242 probable match)
Recipe No: 33873 (1.054992 probable match)
Recipe No: 10051 (1.056956 probable match)
Recipe No: 36251 (1.058794 probable match)
Recipe No: 18735 (1.059071 probable match)
```

------
## References:
* https://www.kaggle.com/rahulsridhar2811/cuisine-classification-with-accuracy-78-88/notebook
* https://towardsdatascience.com/supervised-learning-basics-of-classification-and-main-algorithms-c16b06806cd3
* https://www.quora.com/How-is-the-k-nearest-neighbor-algorithm-different-from-k-means-clustering
* https://datascienceplus.com/k-nearest-neighbors-knn-with-python/
* https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
* https://stackoverflow.com/questions/10695139/sort-a-list-of-tuples-by-2nd-item-integer-value
* https://www.python-course.eu/k_nearest_neighbor_classifier.php
* https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
