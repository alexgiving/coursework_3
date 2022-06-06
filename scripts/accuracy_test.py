from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")

docker_image_name = os.getenv('docker_image_name')


def create(path):
    if not os.path.exists(path):
        os.mkdir(path)


def convert_name(name):
    return name.replace('"', '').replace(":", "").replace("<", "").replace(">", "").replace("|", "").replace("*", "").replace("?", "")


build_path = "build_files/"

filename = f'{build_path}/DecisionTreeClassifier.sav'
my_model = pickle.load(open(filename, 'rb'))


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
    return f"{int(ct/(ct+cf) * 100)}%"


def get_precicion_metrics(acr):
    tp = 0
    fp = 0
    for real, pred in acr:
        if pred == real and pred == 1:
            tp += 1
        elif pred != real and pred == 1:
            fp += 1
    return f"{int(tp/(tp+fp) * 100)}%"


def get_recall_metrics(acr):
    tp = 0
    fn = 0
    for real, pred in acr:
        if pred == real and pred == 1:
            tp += 1
        elif pred != real and pred == 0:
            fn += 1
    return f"{int(tp/(tp+fn) * 100)}%"


my_questions = ["Учащийся",
                "16.Работаете ли Вы?",
                "14.Увлекаетесь ли Вы спортом?",
                "9.Сколько времени Вы уделяете самостоятельной подготовке к занятиям (в среднем)?",
                "8.Как много Вы пропускаете аудиторных занятий?",
                "6.Бывают ли у Вас долги по экзаменам/зачетам?",
                "2.Посещаете ли Вы дополнительные занятия (неважно, в вышке или вне)?",
                "1.Участвуете ли Вы в олимпиадах?"]

my_answers = [
    # 1
    ["287", "Нет", "да, хожу на фитнес или в тренажерный зал",
        "Все свободное время", "Не пропускаете", "1-2 раза в год", "нет", "Нет", "1"],
    # 2
    ["179", "Да", "Другой ответ", "Готовлюсь только перед занятиями",
        "Регулярно пропускаете", "Всегда", "нет", "Нет", "1"],
    # 3
    ["177", "Нет", "нет, не занимаюсь", "От 1 до 3 часов в день",
        "Не пропускаете", "1-2 раза в год", "нет", "Нет", "1"],
    # 4
    ["459", "Нет", "да, занимаюсь командными видами спорта (футбол, баскетбол, хоккей, воллейбол и пр.)",
     "От 1 до 3 часов в день", "Среднее количество пропусков", "1-2 раза в семестр", "нет", "Нет", "0"],
    # 5
    ["415", "Да", "да, хожу на фитнес или в тренажерный зал", "7 часов в неделю", "Среднее количество пропусков",
        "1-2 раза в семестр", "да, по основным предметам моей специальности", "Нет", "1"],
    # 6
    ["463", "Нет", "да, хожу на фитнес или в тренажерный зал", "От 1 до 3 часов в день",
        "Среднее количество пропусков", "1-2 раза в семестр", "нет", "Нет", "0"],
    # 7
    ["367", "Нет", "да, хожу на фитнес или в тренажерный зал", "Готовлюсь только перед занятиями",
        "Среднее количество пропусков", "1-2 раза в семестр", "нет", "Нет", "0"],
    # 8
    ["304", "Нет", "да, хожу на фитнес или в тренажерный зал", "Все свободное время;Более 3 часов в день",
        "Не пропускаете", "1-2 раза в год", "да, по основным предметам моей специальности", "Нет", "1"],
    # 9
    ["464", "Нет", "да, хожу на фитнес или в тренажерный зал", "От 1 до 3 часов в день",
        "Среднее количество пропусков", "Нет", "да, по основным предметам моей специальности", "Нет", "0"],
    # 10
    ["56", "Нет", "да, хожу на фитнес или в тренажерный зал", "От 1 до 3 часов в день",
        "Не пропускаете", "Нет", "да, по основным предметам моей специальности", "Да", "1"]
]

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


print(f"""

<!DOCTYPE html>
<html>""" + """
<style>
table, th, td{
  border: 1px solid black;
}
</style>
""" + """
<body>
<h1 style="color:green"=>The build is done successfully</h1>

<table>
  <tr>
    <th>Metric</th>
    <th>Result</th> 
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>{get_accuracy_metrics(answer)}</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>{get_recall_metrics(answer)}</td> 
  </tr>
 <tr>
    <td>Precicion</td>
    <td>{get_precicion_metrics(answer)}</td> 
  </tr>
</table>  
<a href="https://hub.docker.com/r/alexgiving/{docker_image_name}">Docker image is there</a>
<p>The artifacts is below</p>
</body></html>

""")
