from math import dist
import pandas as pd
import numpy as np

# open dataaset
movies_dataset = pd.read_csv('movies.csv', delimiter=',')


def create_features_mappings(dataset) -> dict:
    """
    Creates dictionary with mapping of movie genres, directors, stars, countries, companies
    To measure distance between points, we need to first transform string types to numeric values
    :argument: dataset
    :return: dictionary with mappings

    Return example
   {
        "directors": {
            "John Smith": 0,
            "Quentin Tarantino": 1
        },
        ...
        "genres": {
            "Action": 0,
            "Adventure": 1
        }
    }

    """
    movie_genres_mapping: dict = create_single_feature_mapping(dataset, 'genre')
    movie_directors_mapping: dict = create_single_feature_mapping(dataset, 'director')
    movie_stars_mapping: dict = create_single_feature_mapping(dataset, 'star')
    movie_countries_mapping: dict = create_single_feature_mapping(dataset, 'country')
    movie_companies_mapping: dict = create_single_feature_mapping(dataset, 'company')

    return {
        "directors": movie_directors_mapping,
        "stars": movie_stars_mapping,
        "countries": movie_countries_mapping,
        "companies": movie_companies_mapping,
        "genres": movie_genres_mapping
    }


def create_single_feature_mapping(dataset: pd.DataFrame, col_name: str) -> dict:
    """
    Create single feature mapping
    To measure distance between points, we need to first transform string types to numeric values
    :argument: dataset, column name
    :return: dictionary with mapping

    Return example
    {
        "Action": 0,
        "Adventure": 1
    }

    """
    mapping = {}
    unique_list = dataset[col_name].array.unique()
    counter = 0
    for feature_name in unique_list:
        mapping[feature_name] = counter
        counter += 1
    return mapping


class KNN:
    """
    Basic KNN algorithm implementation with Euclidean distance as distance metric and k=1 as default value
    """
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        """
        Predicts test data
        Returns list of predicted scores
        :param x_test: list (containing numpyp arrays)
        :return:
        """
        mapping = create_features_mappings(movies_dataset)

        predictions = []
        for row in x_test:
            test_data = row.copy()
            test_data[0] = mapping["genres"][test_data[0]]
            test_data[4] = mapping["directors"][test_data[4]]
            test_data[5] = mapping["stars"][test_data[5]]
            test_data[6] = mapping["countries"][test_data[6]]
            test_data[7] = mapping["companies"][test_data[7]]
            test_data = list(map(lambda e: float(e), test_data))
            label = self.closest_neighbour(test_data)
            predictions.append(label)

        return predictions

    def closest_neighbour(self, row):
        """
        Finds closest neighbour to given data
        :param row:
        :return:
        """
        best_distance = dist(row, self.x_train[0])
        best_index = 0
        for i in range(1, len(self.x_train)):
            # check if there is closer neighbour
            if dist(row, self.x_train[i]) < best_distance:
                best_distance = dist(row, self.x_train[i])
                best_index = i
        return self.y_train[best_index]

# create features and labels lists
features = []
labels = []

# create features mapping from movies dataset
features_mapping = create_features_mappings(movies_dataset)

# append features and labels from dataset to lists
for i in range(len(movies_dataset['score'])):
    features.append(np.array([
        features_mapping["genres"][movies_dataset['genre'][i]],
        int(movies_dataset['year'][i]),
        float(movies_dataset['runtime'][i]),
        float(movies_dataset['votes'][i]),
        features_mapping["directors"][movies_dataset['director'][i]],
        features_mapping["stars"][movies_dataset['star'][i]],
        features_mapping["countries"][movies_dataset['country'][i]],
        features_mapping["companies"][movies_dataset['company'][i]],
    ]))
    labels.append(float(movies_dataset['score'][i]))

# change features and labels lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# create classifier instance
classifier = KNN()

# train classifier
classifier.fit(features, labels)

# examples
print(classifier.predict(
    [
        np.array([
            'Drama',
            1980,
            146.0,
            927000.0,
            'Stanley Kubrick',
            'Jack Nicholson',
            'United Kingdom',
            'Warner Bros.'
        ]),
        np.array([
            'Action',
            1980,
            128.0,
            10000.0,
            'Richard Lester',
            'Jack Nicholson',
            'United Kingdom',
            'Warner Bros.'
        ])

    ]
))
