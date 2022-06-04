from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import pickle
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

build_path = "build_files/"
data = pd.read_excel('src/dataset_2021 -14.xlsx')
data = shuffle(data)
min_mark = 4

data['Сдал'] = pd.cut(x=data['Средний балл'], bins=[
                      0, min_mark, 10], labels=[0, 1])
data.drop(['Средний балл'], axis=1, inplace=True)

data.drop([
    'Дата прохождения теста',
    '15.Образование Ваших родителей?',
    '3.С какими оценками Вы закончили школу?',
    '4.Ходили ли Вы на подготовительные курсы перед поступлением в вуз?',
    '7.Какая у Вас семья?',
    '10.Получали ли Вы стипендию? (в течение последнего года)',
    '11.Оцените, как Вам нравится учиться?',
    '13.На какие средства Вы живете?',
    '5.Брали ли Вы академический отпуск?',
    '17.Укажите Ваше семейное положение.',
    '12.Каковы условия Вашего проживания?'
],
    axis=1, inplace=True)

# That part index all names for privacy reasons
name_hash = []
for idx, name in enumerate(data['Учащийся']):
    name_hash.append((idx, name))
    data.loc[idx, 'Учащийся'] = idx


def create(path):
    if not os.path.exists(path):
        os.mkdir(path)


def train_test_val_split(test_ratio, validation_ratio):
    # Replace all text output to index
    label_encoder = LabelEncoder()
    label_data = data.copy()
    s = (label_data.dtypes == 'object')
    object_cols = list(s[s].index)
    create(build_path)
    for col in object_cols:
        label_encoder.fit(label_data[col])
        label_data[col] = label_encoder.transform(label_data[col])
        file_name = f'{build_path}/{col.replace("/", "-")}_class_linear_encoder.npy'
        f = open(file_name, 'w+')
        np.save(file_name, label_encoder.classes_)
        f.close()

    size = label_data.shape[0]
    validation_ratio = (validation_ratio * size) / (size * (1 - test_ratio))

    X_train, X_test, y_train, y_test = train_test_split(label_data.drop('Сдал', axis=1),
                                                        label_data['Сдал'],
                                                        test_size=test_ratio)

    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=validation_ratio)

    # Output X_train.iloc[:,1:] delete student hash from training and testing selections
    return X_train.iloc[:, 1:], X_test.iloc[:, 1:], X_val, y_train, y_test, y_val


X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(
    test_ratio=0.25, validation_ratio=0.05)

dtr = DecisionTreeClassifier()
dtr.fit(X_train, y_train)

y_pred = dtr.predict(X_test)

filename = f'{build_path}/DecisionTreeClassifier.sav'
pickle.dump(dtr, open(filename, 'wb'))
