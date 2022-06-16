
# # Import Libs


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score,                           \
    r2_score,                           \
    recall_score,                       \
    accuracy_score,                     \
    precision_score,                    \
    mean_squared_error,                 \
    mean_absolute_error,                \
    balanced_accuracy_score,            \
    precision_recall_fscore_support
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Import general functions

# Import lib for saving model and encoder

# Import preprocessing functions

# Import confusion_matrix and roc_auc_curve

# Import metrics functions


build_path = "build_files/"


def convert_name(name):
    # Double quote ", Colon :, Less than <, Greater than >, Vertical bar |, Asterisk *, Question mark ?, Carriage return \r, Line feed \n
    return name.replace('"', '').replace(":", "").replace("<", "").replace(">", "").replace("|", "").replace("*", "").replace("?", "")


# # Preprocessing


# Import dataset
data = pd.read_excel('src/dataset_2021 -14.xlsx')
data = shuffle(data)

# Define minimal mark
min_mark = 4


# Output distribution of scores
plt.scatter(
    range(len(data['Средний балл'])),
    data['Средний балл']
)
plt.grid()


# NOTE: lines in graph illustrate custom increase the number of not pass students.
# That trick were done for better model training. It helps to increase accuracy metricks of models


# Lets find out how many students did not pass the exams


def counter_print():
    passed = 0
    not_passed = 0
    res = 0
    for mark in data['Средний балл']:
        if mark >= min_mark:
            passed += 1
        elif mark < min_mark:
            not_passed += 1
        res += 1
    print(f"TOTAL: {res}\nPassed: {passed}\nNot pas: {not_passed}")


counter_print()


# Replase mark of student by belonging to the class of successfully passed
data['Сдал'] = pd.cut(x=data['Средний балл'], bins=[
                      0, min_mark, 10], labels=[0, 1])
data.drop(['Средний балл'], axis=1, inplace=True)

# Drop indicators which are not connected with extracurricular activities
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


# Dataset with concern questions


# Create a copy of dataset to use it in future work
debug_data = data

data.head()


# Split our data in 3 parts for train, test and validation


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
        # Save lable encouder hash to separate files to use them in future if need
        file_name = f'{build_path}/{col.replace("/", "-")}_class_linear_encoder.npy'
        f = open(file_name, 'w+')
        np.save(file_name, label_encoder.classes_)
        f.close()

# Split dataset
    # NOTE: To fix split selections use random_state= parameter
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


X_train


# # Model variants
# In that part I will choose the classification model which will predict if student pass exams successfully or not


# ## Metrics Functions


def classification_quality(y_test, y_pred):
    print("Accuracy:",          accuracy_score(y_test, y_pred))
    print("Recall:",            recall_score(y_test, y_pred, average='macro'))
    print("Precision:",         precision_score(
        y_test, y_pred, average='macro'))
    print("F1:",                f1_score(y_test, y_pred,
          average='macro', labels=np.unique(y_pred)))
    print("Weighted Recall:",   (precision_recall_fscore_support(
        y_test, y_pred, average='macro')))


# For historical reason
def regression_quality(y_test, y_pred):
    print("MSE:",               mean_squared_error(y_test, y_pred))
    print("RMSE:",              mean_squared_error(y_test, y_pred)**(1/2))
    print("MAE:",               mean_absolute_error(y_test, y_pred))
    print("R2:",                r2_score(y_test, y_pred))


metrics_list = ['Accuracy', 'Balanced Accuracy', 'Recall',
                'Precision', 'F1', 'MSE', 'RMSE', 'MAE', 'R2']
model_array = []
output_array = []


def compilance_print(model, y_test, y_pred, model_flag):
    temp_array = []
    if model_flag == 'cls':  # Classifier
        temp_array.append(accuracy_score(y_test, y_pred))
        temp_array.append(balanced_accuracy_score(y_test, y_pred))
        temp_array.append(recall_score(y_test, y_pred, average='macro'))
        temp_array.append(precision_score(y_test, y_pred, average='macro'))
        temp_array.append(
            f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred)))
        for _ in range(4):
            temp_array.append(None)  # Set regression metrics as None

    # For historical reason
    elif model_flag == 'reg':  # Regression
        for _ in range(5):
            temp_array.append(None)  # Set classifier metrics as None
        temp_array.append(mean_squared_error(y_test, y_pred))
        temp_array.append(mean_squared_error(y_test, y_pred)**(1/2))
        temp_array.append(mean_absolute_error(y_test, y_pred))
        temp_array.append(r2_score(y_test, y_pred))
    else:
        print('Error')
        for _ in metrics_list:
            temp_array.append(None)

    flag = 1
    model_indx = -1
    for indx, _model in enumerate(model_array):
        if _model == model:
            model_indx = indx
            flag = 0
    if flag:
        output_array.append([0] * len(metrics_list))
        model_indx = len(model_array)
        model_array.append(model)
    for indx, el in enumerate(temp_array):
        output_array[model_indx][indx] = el


def graph_show(model, X, y):
    metrics.plot_confusion_matrix(model, X, y)
    metrics.plot_roc_curve(model, X, y)
    plt.show()


# ## Classifier


# ### Perception Classifier Model


Perc = Perceptron()
Perc.fit(X_train, y_train)

y_pred = Perc.predict(X_test)

classification_quality(y_test, y_pred)
compilance_print('Perceptron', y_test, y_pred, 'cls')


# ### Random Forest Classifier Model


RanF = RandomForestClassifier()
RanF.fit(X_train, y_train)

y_pred = RanF.predict(X_test)

classification_quality(y_test, y_pred)
compilance_print('RandomForest', y_test, y_pred, 'cls')


# ### DecisionTreeClassifier Model


dtr = DecisionTreeClassifier()
dtr.fit(X_train, y_train)

y_pred = dtr.predict(X_test)

# plot tree
plt.figure(figsize=(15, 15))
plot_tree(dtr, fontsize=10)

classification_quality(y_test, y_pred)
compilance_print('DecisionTreeClassifier', y_test, y_pred, 'cls')


# Creates dot file named tree.dot
export_graphviz(
    dtr,
    out_file="myTreeName.dot",
    feature_names=list(X_train.columns),
    class_names=['Не Сдал', 'Сдал'],
    filled=True,
    rounded=True)
# Convert dot to png
!dot - Tpng myTreeName.dot - o outfile1.png


# # Example how to return real values from hash to encode decision tree


def get_dict(input):
    df = input.copy()
    total_zip = {}
    for col in pd.DataFrame(df.iloc[:, 1:-1]).columns:
        list_answers = list(set(df[col]))
        encoder = LabelEncoder()
        filename = f"{build_path}/{str(col).replace('/', '-')}_class_linear_encoder.npy"
        encoder.classes_ = np.load(filename, allow_pickle=True)
        list_encouder = encoder.transform(pd.DataFrame(list_answers))
        total_zip[col] = dict(zip(list_answers, list_encouder))
    return total_zip


def get_answer_by_index(_dict, question, index):
    return list(_dict.get(question).keys())[list(_dict.get(question).values()).index(index)]


get_answer_by_index(get_dict(debug_data), "14.Увлекаетесь ли Вы спортом?", 0)


get_dict(debug_data).get("14.Увлекаетесь ли Вы спортом?").get("Другой ответ")


get_dict(debug_data).get("8.Как много Вы пропускаете аудиторных занятий?")


get_dict(debug_data).get(
    "9.Сколько времени Вы уделяете самостоятельной подготовке к занятиям (в среднем)?")


# ## Matrix
# That table helps to understand which model is better for my work


pd.DataFrame(index=model_array, columns=metrics_list, data=output_array).T


# # Model result


my_model = dtr  # My own prefix of DecisionTreeClassifier


# I decided to use DecisionTree Classifier model due to its metrix


# save the model to disk
filename = f'{build_path}/DecisionTreeClassifier.sav'
pickle.dump(my_model, open(filename, 'wb'))

# Load an existed model from disk to use it in product
#my_model = pickle.load(open(filename, 'rb'))


# Output confusion matrix and ROC/AUC curve
graph_show(my_model, X_test, y_test)


# That metrics shows the accuracy of model. As we can see that models return very few False-Positive and True-Negative errors


# # Validation


# Get predicted results of validation selection
# NOTE: Validation selection was not used for training model thus the results are objective
# Also X_val.iloc[:,1:] used for hiding from model hash of students names
y_pred_val = my_model.predict(X_val.iloc[:, 1:])


# These functions allow to output results of validation

def print_with_name():
    print(f'Студент {name_hash[X_val.iloc[i, 0]][1]} предположительно {"сдал(а)" if y_pred_val[i] == 1 else "не сдал(а)"}, в жизни {"сдал(а)" if y_val.iloc[i] == 1 else "не сдал(а)"}')


def print_with_id():
    print(
        f'Студент №{i+1} предположительно {"сдал(а)" if y_pred_val[i] == 1 else "не сдал(а)"}, в жизни {"сдал(а)" if y_val.iloc[i] == 1 else "не сдал(а)"}')


error_counter = 0
total = 0
for i in range(len(y_pred_val)):
    if y_pred_val[i] != y_val.iloc[i] == 1:
        error_counter += 1
    total += 1
    print_with_id()
print(f'There were {error_counter} from {total} error(s)!')

# # Validation with my answers

my_questions = ["Учащийся", "16.Работаете ли Вы?", "14.Увлекаетесь ли Вы спортом?",
                "9.Сколько времени Вы уделяете самостоятельной подготовке к занятиям (в среднем)?",
                "8.Как много Вы пропускаете аудиторных занятий?", "6.Бывают ли у Вас долги по экзаменам/зачетам?",
                "2.Посещаете ли Вы дополнительные занятия (неважно, в вышке или вне)?", "1.Участвуете ли Вы в олимпиадах?"]


my_answers = [
    "Алексей",                                  # Учащийся
    "Да",                                       # 16.Работаете ли Вы?
    "да, хожу на фитнес или в тренажерный зал",  # 14.Увлекаетесь ли Вы спортом?
    # 9.Сколько времени Вы уделяете самостоятельной подготовке к занятиям (в среднем)?
    "Готовлюсь только перед занятиями",
    # 8.Как много Вы пропускаете аудиторных занятий?
    "Не пропускаете",
    # 6.Бывают ли у Вас долги по экзаменам/зачетам?
    "Нет",
    # 2.Посещаете ли Вы дополнительные занятия (неважно, в вышке или вне)?
    "нет",
    "Нет",                                      # 1.Участвуете ли Вы в олимпиадах?
]


df = pd.DataFrame(data=[my_answers], columns=my_questions)

# Iterate by df without name
for col in pd.DataFrame(df.iloc[:, 1:]).columns:
    encoder = LabelEncoder()
    filename = f"{build_path}/{str(col).replace('/', '-')}_class_linear_encoder.npy"
    # Import encouder fit data
    encoder.classes_ = np.load(filename, allow_pickle=True)
    df[col] = encoder.transform(df[col])


y_pred_my = my_model.predict(df.iloc[:, 1:])                 # Without name
print(
    f'{df.iloc[0,0]} предположительно {"сдал(а)" if y_pred_my == 1 else "не сдал(a)"} экзамен')

# # Correlation matrix


def exists(path):
    try:
        os.stat(path)
    except OSError:
        return False
    return True


# Replace all text output to index
# Used for outputing correlation matrix
label_encoder = LabelEncoder()
label_data = data.iloc[:, 1:].copy()

for col in label_data.columns:
    filename = f"{build_path}/{col.replace('/', '-')}_class_linear_encoder.npy"
    if exists(filename):
        label_encoder.classes_ = np.load(filename, allow_pickle=True)
    else:
        label_encoder.fit(data[col])
#   always:
    label_data[col] = label_encoder.transform(label_data[col])


# Creating correlation matrix
rs = np.random.RandomState(0)
corr = label_data.corr()
corr.style.background_gradient(cmap='coolwarm', axis=None)

# # Example of product usage

filename = f'{build_path}/DecisionTreeClassifier.sav'
my_model = pickle.load(open(filename, 'rb'))

my_questions = ["Учащийся",
                "16.Работаете ли Вы?",
                "14.Увлекаетесь ли Вы спортом?",
                "9.Сколько времени Вы уделяете самостоятельной подготовке к занятиям (в среднем)?",
                "8.Как много Вы пропускаете аудиторных занятий?",
                "6.Бывают ли у Вас долги по экзаменам/зачетам?",
                "2.Посещаете ли Вы дополнительные занятия (неважно, в вышке или вне)?",
                "1.Участвуете ли Вы в олимпиадах?"]

my_answers = []

for q in my_questions:
    my_answers.append(str(input(f'Введите ответ на вопрос "{q}": ')))

df = pd.DataFrame(data=[my_answers], columns=my_questions)

# Iterate by df without name
for col in pd.DataFrame(df.iloc[:, 1:]).columns:
    encoder = LabelEncoder()
    filename = f"{build_path}/{str(col).replace('/', '-')}_class_linear_encoder.npy"
    # Import encouder fit data
    encoder.classes_ = np.load(filename, allow_pickle=True)
    df[col] = encoder.transform(df[col])


y_pred_my = my_model.predict(df.iloc[:, 1:])                 # Without name
print(
    f'{df.iloc[0,0]} предположительно {"сдал(а)" if y_pred_my == 1 else "не сдал(а)"} экзамен')
