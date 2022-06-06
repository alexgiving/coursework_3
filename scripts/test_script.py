
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
filename = f'{build_path}/DecisionTreeClassifier.sav'


def convert_name(name):
    # Double quote ", Colon :, Less than <, Greater than >, Vertical bar |, Asterisk *, Question mark ?, Carriage return \r, Line feed \n
    return name.replace('"', '').replace(":", "").replace("<", "").replace(">", "").replace("|", "").replace("*", "").replace("?", "")


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

name_hash = []
for idx, name in enumerate(data['Учащийся']):
    name_hash.append((idx, name))
    data.loc[idx, 'Учащийся'] = idx


my_model = pickle.load(open(filename, 'rb'))


my_questions = ["Учащийся",
                "16.Работаете ли Вы?",
                "14.Увлекаетесь ли Вы спортом?",
                "9.Сколько времени Вы уделяете самостоятельной подготовке к занятиям (в среднем)?",
                "8.Как много Вы пропускаете аудиторных занятий?",
                "6.Бывают ли у Вас долги по экзаменам/зачетам?",
                "2.Посещаете ли Вы дополнительные занятия (неважно, в вышке или вне)?",
                "1.Участвуете ли Вы в олимпиадах?"]


def train_test_val_split():
    # Replace all text output to index
    label_encoder = LabelEncoder()
    label_data = data.copy()
    s = (label_data.dtypes == 'object')
    object_cols = list(s[s].index)
    for col in object_cols:
        filename = f"{build_path}/{convert_name(str(col)).replace('/', '-')}_class_linear_encoder.npy"
        label_encoder.classes_ = np.load(filename, allow_pickle=True)
        label_data[col] = label_encoder.transform(label_data[col])
    X_train, X_test, y_train, y_test = train_test_split(label_data.drop('Сдал', axis=1),
                                                        label_data['Сдал'])

    return X_train.iloc[:, 1:], X_test.iloc[:, 1:], y_train, y_test


X_train, X_test, y_train, y_test = train_test_val_split()

y_pred = my_model.predict(X_test)

error_counter = 0
total = 0
for i in y_pred:
    for j in y_test:
        total += 1
        if int(i) != int(j):
            error_counter += 1

print(
    f'There were {error_counter} from {total} error(s)! \nAccuracy is {(total - error_counter )/ total}')
