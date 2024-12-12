
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from random import randint

# Load the dataset
data = pd.DataFrame({
    "Year": [
        1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974,
        1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
        1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004,
        2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
        2020, 2021, 2022, 2023
    ],
    "Favored Number": [
        30, 45, 92, 62, 15, 28, 82, 16, 30, 3, 94, 54, 89, 33, 31, 19, 51, 52, 50, 74, 44, 4, 96, 15,
        17, 30, 40, 94, 59, 7, 37, 50, 25, 91, 75, 39, 8, 65, 72, 6, 9, 83, 33, 20, 62, 3, 40, 61, 66,
        90, 41, 25, 70, 67, 29, 63, 93, 6, 19, 15, 66, 19, 0, 94
    ]
})

# Split the data into features (X) and target (y)
X = data[['Year']]
y = data['Favored Number']

# Train a Random Forest Regression model
model = RandomForestRegressor(n_estimators=1000, random_state=42)
model.fit(X, y)

# Generate predictions for 10 iterations
for i in range(7):
    # Generate a random year for prediction
    new_data = pd.DataFrame({
        "Year": [randint(2024, 2024)]  # Random years within the range of the dataset
    })

    # Predict the favored number for the random year
    prediction = model.predict(new_data)[0]

    # Print the result
    print(f"{i+1:02d}. Predicted favored number for year {new_data['Year'].iloc[0]} is: {round(prediction)}")
