from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")


def create(path):
    if not os.path.exists(path):
        os.mkdir(path)


def convert_name(name):
    return name.replace('"', '').replace(":", "").replace("<", "").replace(">", "").replace("|", "").replace("*", "").replace("?", "")


build_path = "build_files/"

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

my_answers = [
    ["287", "Нет", "да, хожу на фитнес или в тренажерный зал",
        "Все свободное время", "Не пропускаете", "1-2 раза в год", "нет", "Нет", "1"],
    ["179", "Да", "Другой ответ", "Готовлюсь только перед занятиями",
        "Регулярно пропускаете", "Всегда", "нет", "Нет", "1"],
    ["177", "Нет", "нет, не занимаюсь", "От 1 до 3 часов в день",
        "Не пропускаете", "1-2 раза в год", "нет", "Нет", "1"],
    ["459", "Нет", "да, занимаюсь командными видами спорта (футбол, баскетбол, хоккей, воллейбол и пр.)",
     "От 1 до 3 часов в день", "Среднее количество пропусков", "1-2 раза в семестр", "нет", "Нет", "0"],
    ["415", "Да", "да, хожу на фитнес или в тренажерный зал", "7 часов в неделю", "Среднее количество пропусков",
        "1-2 раза в семестр", "да, по основным предметам моей специальности", "Нет", "1"]

]


def print_with_name(i):
    print(
        f'Студент {i} предположительно {"сдал(а)" if int(y_pred_my) == 1 else "не сдал(а)"}, в жизни {"сдал(а)" if int(my_answers[i][-1]) == 1 else "не сдал(а)"}')


def get_accuracy_metrics(acr):
    ct = 0
    cf = 0
    for real, pred in acr:
        if pred == real:
            ct += 1
        else:
            cf += 1
    return f"Accuracy of model: {ct/(ct+cf) * 100}%"


answer = []
for i in range(len(my_answers)):
    df = pd.DataFrame(data=[my_answers[i][:-1]], columns=my_questions)

    for col in pd.DataFrame(df.iloc[:, 1:]).columns:
        encoder = LabelEncoder()
        filename = f"{build_path}/{convert_name(str(col)).replace('/', '-')}_class_linear_encoder.npy"
        encoder.classes_ = np.load(filename, allow_pickle=True)
        df[col] = encoder.transform(df[col])

    y_pred_my = my_model.predict(df.iloc[:, 1:])

    answer.append((int(my_answers[i][-1]), int(y_pred_my)))
    # print_with_name(i)


print(f"""  <!DOCTYPE html>
            <html>
            <body>
            <h1>The build is done</h1>
            <p>
            {get_accuracy_metrics(answer)}
            </p>
            <p>The artifacts is below</p>
            </body>
            </html>""")
