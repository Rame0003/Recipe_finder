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
### To run the code:
Please use the project3.ipynb file. The only inputs from the user side are 
1. The file location
2. User ingredients
3. Number of recipes
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

For my example, I have taken k=11 after using the accuracy measure using the accuracy score module from the sklearn package. I used the following code to obtain the result:
```python
for K in range(22):
    K_value = K+1
    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto', metric='minkowski')
    neigh.fit(X_train,y_train)
    y_pred = neigh.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",K_value)
    
```
Given below are the results:
```output
Accuracy is  58.75 % for K-Value: 1
Accuracy is  53.75 % for K-Value: 2
Accuracy is  59.0 % for K-Value: 3
Accuracy is  59.5 % for K-Value: 4
Accuracy is  60.75000000000001 % for K-Value: 5
Accuracy is  61.0 % for K-Value: 6
Accuracy is  61.75000000000001 % for K-Value: 7
Accuracy is  62.0 % for K-Value: 8
Accuracy is  62.25000000000001 % for K-Value: 9
Accuracy is  64.0 % for K-Value: 10
**Accuracy is  64.5 % for K-Value: 11**
Accuracy is  63.24999999999999 % for K-Value: 12
Accuracy is  62.74999999999999 % for K-Value: 13
Accuracy is  62.74999999999999 % for K-Value: 14
Accuracy is  62.25000000000001 % for K-Value: 15
Accuracy is  63.0 % for K-Value: 16
Accuracy is  63.24999999999999 % for K-Value: 17
Accuracy is  63.24999999999999 % for K-Value: 18
Accuracy is  62.5 % for K-Value: 19
Accuracy is  63.24999999999999 % for K-Value: 20
Accuracy is  61.75000000000001 % for K-Value: 21
Accuracy is  61.0 % for K-Value: 22
```
We can see that the  accuracy for a k-value of 11 is higher than the other values. Hence, I have decided to take 11 as the k-value.

## Display results:

Once the classification has been performed, the test data is provided and the prediction is performed. We can provide the test ingredient vector matrix, the trained model and the number of ingredients that we want to the function. The input to display the results are as follows:
````python
get_results(test, model, n)
````
With the above function, we obtain the results according to the given user input. Care should be taken that the number of recipes are limited to the K-value provided. 
```output
The model predicts that the ingredients resembles indian (54.545455 resemblence)

Recipe No: 37573 (0.882196 probable match)
Recipe No: 29693 (0.974139 probable match)
Recipe No: 13518 (1.005656 probable match)
Recipe No: 20178 (1.009164 probable match)
Recipe No: 11906 (1.057354 probable match)
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
