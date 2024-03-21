# California Housing Prices SLR and MLR

This dataset contains median home prices for California counties as obtained at the 1990 census. The dataset used in this repo was obtained from Kaggle. The link to the relevant dataset and my Kaggle profile link are below. If you wish, you can follow me and my data science studies. 

Link to the dataset: [Click here](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

My Kaggle profile link: [Click here](https://www.kaggle.com/senacetinkaya)

## About Dataset

This is the dataset used in the second chapter of Aurélien Géron's latest book 'Applied Machine Learning with Scikit-Learn and TensorFlow'. It is an excellent introduction to implementing machine learning algorithms as it requires basic data cleaning and has an easily understandable list of variables. The columns in the data set are as follows:

- longitude
- latitude
- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income
- median_house_value
- ocean_proximity

## Libraries used in the project

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
```

### You can access the data analysis graphics created within the scope of the project from the images in this repo.
