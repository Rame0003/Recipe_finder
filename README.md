# CS 5293 Spring 2020 Project 3
## Cuisine/Recipe Finder
### By Abilash Ramesh
-------
## Contents:
* Build dataframe / data cleansing
* Feature extraction from dataset
* Data classification
* Display results
* Testcases 
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


## Feature extraction from dataset:

We will begin the process of feature extraction by 

------
## References:
* https://www.kaggle.com/rahulsridhar2811/cuisine-classification-with-accuracy-78-88/notebook
* https://towardsdatascience.com/supervised-learning-basics-of-classification-and-main-algorithms-c16b06806cd3
* https://www.quora.com/How-is-the-k-nearest-neighbor-algorithm-different-from-k-means-clustering
* https://datascienceplus.com/k-nearest-neighbors-knn-with-python/
* https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
