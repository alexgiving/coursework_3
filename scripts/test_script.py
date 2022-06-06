
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


docker_image_name = os.getenv('docker_image_name')
repo_name = os.getenv('repo_name')
sha_git = os.getenv('sha_git')


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


def get_accuracy_metrics(a, b):
    ct = 0
    cf = 0
    for real in a:
        for pred in b:
            if pred == real:
                ct += 1
            else:
                cf += 1
    return f"{int(ct/(ct+cf) * 100)}%"


def get_precicion_metrics(a, b):
    tp = 0
    fp = 0
    for real in a:
        for pred in b:
            if pred == real and pred == 1:
                tp += 1
            elif pred != real and pred == 1:
                fp += 1
    return f"{int(tp/(tp+fp) * 100)}%"


def get_recall_metrics(a, b):
    tp = 0
    fn = 0
    for real in a:
        for pred in b:
            if pred == real and pred == 1:
                tp += 1
            elif pred != real and pred == 0:
                fn += 1
    return f"{int(tp/(tp+fn) * 100)}%"


print("""

<!DOCTYPE html>
<html>""" + """
<style>
table, th, td{
  border: 1px solid black;
}
</style>
""" + f"""
<body>
<h1 style="color:green">The build is done successfully</h1>

<a href="https://github.com/{repo_name}/commit/{sha_git}">See changes</a>

<table>
  <tr>
    <th>Metric</th>
    <th>Result</th> 
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>{get_accuracy_metrics(y_pred, y_test)}</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>{get_recall_metrics(y_pred, y_test)}</td> 
  </tr>
 <tr>
    <td>Precicion</td>
    <td>{get_precicion_metrics(y_pred, y_test)}</td> 
  </tr>
</table>  
<a href="https://hub.docker.com/r/alexgiving/{docker_image_name}">Docker image is there</a>
</body></html>

""")
