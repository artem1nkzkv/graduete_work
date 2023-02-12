#подгрузка необходимых библиотек

import os #для запуска скриптов
import numpy as np #математические вычисления
import pandas as pd #библиотека для обработки и анализа структурированных данных
import tkinter as tk #графический интерфейс
from tkinter import messagebox
from tkinter.filedialog import askopenfilename #для выбора тестового набора данных
from tkinter.ttk import Progressbar, Style #вывод прогресса выполнения
from tipsDisplay import * #подсказки на экране, импортируется из отдельного файла
from sklearn.model_selection import train_test_split #для разделения набора данных на обучающие и тестовые данные
from sklearn.preprocessing import StandardScaler #для масштабирования набора данных
from sklearn.neighbors import KNeighborsClassifier #метод k-ближайших соседей
from sklearn.metrics import accuracy_score #для поиска точности модели
from sklearn.tree import DecisionTreeClassifier #дерево принятий решений
from sklearn.neural_network import MLPClassifier  #алгоритм многослойного перцептрона
from sklearn.ensemble import RandomForestClassifier #метод случайного леса
from sklearn.metrics import f1_score #поиск F1-score для модели

#размеры окна
HEIGHT = 600
WIDTH = 800
    
#изменение состояния кнопок обычное/отключенное
def buttons_state(state):
    for button in buttons:
        button['state'] = state

#сбросить индикатор выполнения и очистить холст
def clean_window():
    result['text'] = ""
    progressBar['value'] = 0
    s.configure("LabeledProgressbar", text="")
    update_ui()

def update_ui():
    root.update_idletasks()
    root.update()

#создания нового захвата пакетов
def new_capture():
    clean_window()
    os.system('sudo -A python3 packageCollect.py')

#выбор тестового набора данных
def take_database():
    filename = askopenfilename(title="Выберите тестовый набор данных", filetypes=(("CSV Files","*.csv"),))
    return filename

#анализ пакетов vpn
def new_vpn_test():
    buttons_state('disabled')
    result['text'] = ""
    progressBar['value'] = 0
    s.configure("LabeledProgressbar", text="Analysed: {0}%".format(0))
    update_ui()

    #подгружаем базу данных
    try:
        vpnData = pd.read_csv('sampleDatabase/vpnDatabase.csv')
    except Exception as e:
        tk.messagebox.showerror("Error: ", e)
        buttons_state('normal')
        return

    testFileName = take_database()

    if not testFileName:
        buttons_state('normal')
        return
    try:
        testData = pd.read_csv(testFileName) #загрузка .csv файла с данными  
    except Exception as e:
        tk.messagebox.showerror("Error: ", e)
        buttons_state('normal')
        return
    
    #устанавливаем необходимые нам колонки для последующей проверки
    cols = ['Version', 'Protocol', 'TTL', 'SrcAddress', 'DestAddress', 'SrcPort', 'DestPort', 'SeqNum', 'AckNum', 'Flag', 'DataSize', 'Service']
    test_cols = list(testData.columns.values)

    #если неверный формат базы данных
    if cols != test_cols:
        tk.messagebox.showerror("Неверный набор данных","Набор данных не соответствует требуемым спецификациям!")
        buttons_state('normal')
        return

    #если база данных пустая
    if testData.empty:
        tk.messagebox.showerror("Пустой набор данных", "Набор данных пуст. Убедитесь, что вы правильно захватили пакеты.")
        buttons_state('normal')
        return

    #если база данных больше установленного размера
    dataset_size_limit = 700000	#в байтах
    if os.stat(testFileName).st_size > dataset_size_limit:
        if not messagebox.askyesno("Большой набор данных","Набор данных очень большого размера. Расчет может занять несколько минут, используя множество системных ресурсов. Вы уверены что хотите продолжить?"):
            buttons_state('normal')
            return

    #удаляем ненужные столбцы
    drop_cols = ['Version', 'Protocol', 'SrcAddress', 'DestAddress']

    vpnData = vpnData.drop(drop_cols,1) #1- удалить метки из столбцов
    testData = testData.drop(drop_cols,1)

    #удаление дубликатов в наборе данных
    vpnData = vpnData.drop_duplicates()

    #отображение прогресса
    progressBar['value'] = 10
    s.configure("LabeledProgressbar", text="Analysed: {0}%".format(10))
    update_ui()

    X = vpnData.iloc[:,:-1] #выбираем ячейки набора данных
    y = vpnData.iloc[:,-1]

    X_testSet = testData.iloc[:,:]

    #Использование One Hot Encoding (быстрое кодирование) для кодирования строковых данных в числовые
    oneHotFeatures = ['Flag','Service']

    def encode_and_bind(original_dataframe, feature_to_encode):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
        res = pd.concat([original_dataframe, dummies], axis=1)
        res = res.drop([feature_to_encode], axis=1)
        return(res)

    frames = [X, X_testSet]
    temp = pd.concat(frames)

    for feature in oneHotFeatures:
        temp = encode_and_bind(temp, feature)

    X = temp.iloc[0:len(X),:]
    X_testSet = temp.iloc[-len(X_testSet):,:]

    progressBar['value'] = 20
    s.configure("LabeledProgressbar", text="Analysed: {0}%".format(20))

    update_ui()

    #разделить данные на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

    #масштабирование данных
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_testSet = scaler.transform(X_testSet)

    progressBar['value'] = 30
    s.configure("LabeledProgressbar", text="Проанализировано: {0}%".format(30))
    update_ui()

    #метод k-ближайших соседей
    knn = KNeighborsClassifier(n_neighbors=15,metric='minkowski',p=2)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    y_predTest = knn.predict(X_testSet)

    result['text'] = '\n------------------------'
    result['text'] += '\nKNN-классификатор'

    result['text'] += "\nТочность модели на обучающем наборе данных: " + str(accuracy_score(y_test,y_pred))
    knn_prob = sum(y_predTest)/len(y_predTest)
    result['text'] += "\nИспользование VPN: " + str(knn_prob)

    progressBar['value'] = 50
    s.configure("LabeledProgressbar", text="Проанализировано: {0}%".format(50))
    update_ui()

    #дерево принятий решений
    DTC = DecisionTreeClassifier(random_state=0)
    model = DTC.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    y_predTest = model.predict(X_testSet)

    result['text'] += '\n\n------------------------'
    result['text'] += '\nДерево принятия решений'

    result['text'] += "\nТочность модели на обучающем наборе данных: " + str(accuracy_score(y_test,y_pred))
    dt_prob = (sum(y_predTest)/len(y_predTest))
    result['text'] += "\nИспользование VPN: " + str(dt_prob)

    progressBar['value'] = 60
    s.configure("LabeledProgressbar", text="Проанализировано: {0}%".format(60))
    update_ui()

   #алгоритм многослойного перцептрона
    clf = MLPClassifier(max_iter=1500,random_state=1)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_predTest = clf.predict(X_testSet)

    result['text'] += '\n\n------------------------'
    result['text'] += '\nMLP-классификатор'

    result['text'] += "\nМодель F1-score: " + str(f1_score(y_test,y_pred))
    mlp_prob = (sum(y_predTest)/len(y_predTest))
    result['text'] += "\nИспользование VPN: " + str(mlp_prob)
    progressBar['value'] = 70
    s.configure("LabeledProgressbar", text="Проанализировано: {0}%".format(70))
    update_ui()

    #метод случайного леса
    clf = RandomForestClassifier(max_depth=15, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_predTest = clf.predict(X_testSet)
    result['text'] += '\n\n------------------------'
    result['text'] += '\nМетод случайного леса'

    result['text'] += "\nТочность модели на обучающем наборе данных: " + str(accuracy_score(y_test,y_pred))
    rfc_prob = (sum(y_predTest)/len(y_predTest))
    result['text'] += "\nИспользование VPN: " + str(rfc_prob)
    progressBar['value'] = 100
    s.configure("LabeledProgressbar", text="Проанализировано: {0}%".format(100))
    update_ui()

    tk.messagebox.showinfo("Проверка VPN", "Использование VPN: {:.2%}".format(((knn_prob+dt_prob+mlp_prob+rfc_prob)/4.0)))
    buttons_state('normal')


#анализ пакетов proxy
def new_proxy_test():
    buttons_state('disabled')
    result['text'] = ""
    progressBar['value'] = 0
    s.configure("LabeledProgressbar", text="Проанализировано: {0}%".format(0))
    update_ui()

    #подгружаем базу данных
    try:
        proxyData = pd.read_csv('sampleDatabase/proxyDatabase.csv')
    except Exception as e:
        tk.messagebox.showerror("Error: ", e)
        buttons_state('normal')
        return

    testFileName = take_database()

    if not testFileName:
        buttons_state('normal')
        return
    try:
        testData = pd.read_csv(testFileName) #загрузка .csv файла с данными  
    except Exception as e:
        tk.messagebox.showerror("Error: ", e)
        buttons_state('normal')
        return

    #устанавливаем необходимые нам колонки для последующей проверки
    cols = ['Version', 'Protocol', 'TTL', 'SrcAddress', 'DestAddress', 'SrcPort', 'DestPort', 'SeqNum', 'AckNum', 'Flag', 'DataSize', 'Service']
    test_cols = list(testData.columns.values)

    #если неверный формат базы данных
    if cols != test_cols:
        tk.messagebox.showerror("Неверный набор данных","Набор данных не соответствует требуемым спецификациям!")
        buttons_state('normal')
        return

    #если база данных пустая
    if testData.empty:
        tk.messagebox.showerror("Пустой набор данных", "Набор данных пуст. Убедитесь, что вы правильно захватили пакеты.")
        buttons_state('normal')
        return

    #если база данных больше установленного размера
    dataset_size_limit = 700000	#в байтах
    if os.stat(testFileName).st_size > dataset_size_limit:
        if not messagebox.askyesno("Большой набор данных","Набор данных очень большого размера. Расчет может занять несколько минут, используя множество системных ресурсов. Вы уверены что хотите продолжить?"):
            buttons_state('normal')
            return

    #отображение прогресса   
    progressBar['value'] = 10
    s.configure("LabeledProgressbar", text="Проанализировано: {0}%".format(10))
    update_ui()

    #удаляем ненужные столбцы
    drop_cols = ['Version', 'Protocol', 'SrcAddress', 'DestAddress']

    proxyData = proxyData.drop(drop_cols,1)
    testData = testData.drop(drop_cols,1)

    #удаление дубликатов в наборе данных
    proxyData = proxyData.drop_duplicates()

    progressBar['value'] = 20
    s.configure("LabeledProgressbar", text="Проанализировано: {0}%".format(20))
    update_ui()

    X = proxyData.iloc[:,:-1] #выбираем ячейки набора данных
    y = proxyData.iloc[:,-1]

    X_testSet = testData.iloc[:,:]

    #Использование One Hot Encoding (быстрое кодирование) для кодирования строковых данных в числовые
    oneHotFeatures = ['Flag','Service']

    def encode_and_bind(original_dataframe, feature_to_encode):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
        res = pd.concat([original_dataframe, dummies], axis=1)
        res = res.drop([feature_to_encode], axis=1)
        return(res)

    frames = [X, X_testSet]
    temp = pd.concat(frames)

    for feature in oneHotFeatures:
        temp = encode_and_bind(temp, feature)

    X = temp.iloc[0:len(X),:]
    X_testSet = temp.iloc[-len(X_testSet):,:]

    progressBar['value'] = 30
    s.configure("LabeledProgressbar", text="Проанализировано: {0}%".format(30))
    update_ui()

    #разделить данные на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

    #масштабирование данных
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_testSet = scaler.transform(X_testSet)

    progressBar['value'] = 40
    s.configure("LabeledProgressbar", text="Проанализировано: {0}%".format(40))
    update_ui()

    #метод k-ближайших соседей
    knn = KNeighborsClassifier(n_neighbors=7,metric='minkowski',p=2)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    y_predTest = knn.predict(X_testSet)

    result['text'] = '\n\n------------------------'
    result['text'] += '\nKNN-классификатор'

    result['text'] += "\nТочность модели на обучающем наборе данных: " + str(accuracy_score(y_test,y_pred))
    knn_prob = sum(y_predTest)/len(y_predTest)
    result['text'] += "\nИспользование proxy: " + str(knn_prob)

    progressBar['value'] = 70
    s.configure("LabeledProgressbar", text="Проанализировано: {0}%".format(70))
    update_ui()

   #алгоритм многослойного перцептрона
    clf = MLPClassifier(max_iter=1500,random_state=1)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_predTest = clf.predict(X_testSet)

    result['text'] += "\n\n------------------------"
    result['text'] += '\nMLP Classifier'

    result['text'] += "\nМодель F1-score: " + str(f1_score(y_test,y_pred))
    mlp_prob = sum(y_predTest)/len(y_predTest)
    result['text'] += "\nИспользование proxy: " + str(mlp_prob)
    progressBar['value'] = 100
    s.configure("LabeledProgressbar", text="Проанализировано: {0}%".format(100))
    update_ui()
    tk.messagebox.showinfo("Проверка proxy", "Использование proxy: {:.2%}".format(((knn_prob+mlp_prob)/2.0)))
    buttons_state('normal')

#создание окна и заголовка
root = tk.Tk()
root.title("Программное средство, осуществляющее детектирование потоков VPN трафика")
root.option_add('*Dialog.msg.font', 'Helvetica 12')
x = (root.winfo_screenwidth() - root.winfo_reqwidth()) / 3
y = (root.winfo_screenheight() - root.winfo_reqheight()) / 4
root.wm_geometry("+%d+%d" % (x, y))

canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

#добавляем фоновое изображение
background_img = tk.PhotoImage(file='images/background.png')
background_label = tk.Label(root, image=background_img)
background_label.place(relwidth=1, relheight=1)

#кнопка для проверки proxy
proxyTestButton = tk.Button(root, text="PROXY", font=('Helvetica', 12), bd=4, fg='black',bg='white', command=lambda: new_proxy_test())
proxyTestButton.place(relx=0.05, rely=0.2, relwidth=0.2, relheight=0.1)
CreateToolTip(proxyTestButton,text="Проверьте, использовался ли прокси-сервер для файла захвата пакетов .csv")

#кнопка для проверки vpn
vpnTestButton = tk.Button(root, text="VPN", font=('Helvetica',12), bd=4, fg='black',bg='white', command=lambda: new_vpn_test())
vpnTestButton.place(relx=0.05, rely=0.4,  relwidth=0.2, relheight=0.1)
CreateToolTip(vpnTestButton,text="Проверьте, использовался ли VPN в файле захвата пакетов .csv")

#создание рамки для отображения результатов
lower_frame = tk.Frame(root, bg='black', bd=10)
lower_frame.place(relx=0.65, rely=0.15, relwidth=0.55, relheight=0.6, anchor='n')

#добавляем полосу прогресса
s = Style(lower_frame)
s.layout("LabeledProgressbar",
         [('LabeledProgressbar.trough',
           {'children': [('LabeledProgressbar.pbar',
                          {'side': 'left', 'sticky': 'ns'}),
                         ("LabeledProgressbar.label",
                          {"sticky": ""})],
           'sticky': 'nswe'})])
s.configure("LabeledProgressbar", foreground='black', background='#00FF00')
progressBar = Progressbar(lower_frame, orient='horizontal', length=100, mode='determinate', style="LabeledProgressbar")
progressBar.place(relwidth=1)

#показываем окончательный результат
result = tk.Label(lower_frame, font=('Courier',8), anchor='nw', justify='left', bd=4)
result.place(rely=0.1, relwidth=1, relheight=0.9)

#кнопка для очистки холста
clearButton = tk.Button(root, text="Очистка", bd=4, font=('Helvetica',10), fg='white',bg='black', command=lambda:clean_window())
clearButton.place(relx=0.6,rely=0.8, relwidth=0.1, relheight=0.05)
CreateToolTip(clearButton,text="Очистить записи на холсте")

#кнопка для создания нового файла тестового захвата
newCaptureButton = tk.Button(root, text="Захват данных", bd=4, font=('Helvetica',12), fg='black', bg='white', command=lambda:new_capture())
newCaptureButton.place(relx=0.05, rely=0.6, relwidth=0.2, relheight=0.1)

#переменная, содержащая все переменные кнопки
buttons = [proxyTestButton, vpnTestButton, clearButton, newCaptureButton]

#закрытие программы
def on_closing():
    if messagebox.askyesno("Выход", "Вы уверены, что хотите выйти?"):
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
