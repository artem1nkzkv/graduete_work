#импортируем необходимые библиотеки
import socket #модуль для работы с сокетами
import csv #модуль для работы с csv файлами
from struct import * #используется для создания и вытягивания упакованных двоичных данных из строк
import tkinter as tk #графический интерфейс
from threading import Thread #работа с потоками
from pathvalidate import ValidationError, validate_filename #библиотека для очистки/проверки строки
import tkinter.simpledialog as simpledialog  #создания простых модальных диалогов для получения значения от пользователя.
import platform #инструменты для получения сведений об аппаратной платформе, операционной системе и интерпретаторе на которой выполняется программа.
from tipsDisplay import * #импортируем файлы с подсказками

#функция сбора пакетов
def packet_capture(test_file):
	c = True
	#Создание RAW сокета
	try:
		s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP)
	except socket.error as e:
		tk.messagebox.showerror("Socket Error",e)
		c = False
	except Exception as e:
		tk.messagebox.showerror("Error!!!",e)
		c = False
	finally:
		if not c:
			startButton['state'] = 'normal'
			stopButton['state'] = 'disabled'
			label['text'] = ""
			return

	#создаем файл csv для хранения захваченных пакетов
	outputFile = open('collectedDatabase/'+test_file,'w',newline='')
	writer = csv.writer(outputFile)

	#записываем заголовочную строку
	writer.writerow(['Version', 'Protocol', 'TTL', 'SrcAddress', 'DestAddress', 'SrcPort', 'DestPort', 'SeqNum', 'AckNum', 'Flag', 'DataSize', 'Service'])

	#получаем пакеты
	while True:
		if stop == 1:
			label['text'] = ""
			startButton['state'] = 'normal'
			break
		packet = s.recvfrom(65565)
		
		#переводим содержимое кортежа в строковый тип
		packet = packet[0]
		
		#берем первые 20 байт для IP заголовка 
		ip_header = packet[0:20]
		
		#распаковываем из байтового формата
		iph = unpack('!BBHHHBBH4s4s', ip_header)
		version_ihl = iph[0]
		version = version_ihl >> 4
		ihl = version_ihl & 0xF
		iph_length = ihl * 4
		ttl = iph[5]
		protocol = iph[6]
		s_addr = socket.inet_ntoa(iph[8])
		d_addr = socket.inet_ntoa(iph[9])
		
		#TCP заголовок начинается сразу после заголовка IP и обычно имеет длину 20 байт
		tcp_header = packet[20:40]
		
		#распаковываем из байтового формата
		tcph = unpack('!HHLLBBHHH', tcp_header)
		source_port = tcph[0]
		dest_port = tcph[1]
		sequence = tcph[2]
		ack = tcph[3]
		doff_reserved = tcph[4]
		tcph_length = doff_reserved >> 4
		h_size = iph_length + tcph_length * 4
		data_size = len(packet) - h_size
		
		#Выбераем байты, содержащие флаги TCP
		tcpFlag = packet[33:34].hex()
		
		if tcpFlag == "01":
			Flag = "FIN"
		elif tcpFlag == "02":
			Flag = "SYN"
		elif tcpFlag == "03":
			Flag = "FIN-SYN"
		elif tcpFlag == "08":
			Flag = "PSH"
		elif tcpFlag == "09":
			Flag = "FIN-PSH"
		elif tcpFlag == "0A":
			Flag = "SYN-PSH"
		elif tcpFlag == "10":
			Flag = "ACK"
		elif tcpFlag == "11":
			Flag = "FIN-ACK"
		elif tcpFlag == "12":
			Flag = "SYN-ACK"
		elif tcpFlag == "18":
			Flag = "PSH-ACK"
		else:
			Flag = "OTH"
		
		#Выбираем HTTP/HTTPS пакеты
		if source_port == 80 or source_port == 443:
			if source_port == 80:
				writer.writerow([str(version), str(protocol), str(ttl), str(s_addr), str(d_addr), str(source_port), str(dest_port), str(sequence), str(ack), Flag, str(data_size), "HTTP"])
				print("Данные захвачены")
			else:
				writer.writerow([str(version), str(protocol), str(ttl), str(s_addr), str(d_addr), str(source_port), str(dest_port), str(sequence), str(ack), Flag, str(data_size), "HTTPS"])
				print("Данные захвачены")

	#после завершения захвата данных
	outputFile.close()
	tk.messagebox.showinfo("Захват завершен", "Тестовый захват завершен. Теперь вы можете закрыть окно")

#создание потока для захвата пакетов
def start_capture():
	startButton['state'] = 'disabled'#cостояние кнопок
	stopButton['state'] = 'normal'
	#запрос имени файла
	test_file_name = simpledialog.askstring(title="File Name", prompt="Введите имя тестового файла (без расширения .csv): ", initialvalue='test')
	if test_file_name is None:
		startButton['state'] = 'normal'
		stopButton['state'] = 'disabled'
		return
	try:
		validate_filename(test_file_name, platform=platform.system()) #подтвердить имя файла для текущей ОС
	except ValidationError as e:
		tk.messagebox.showerror("Неверное имя файла", "Введенное имя файла неверно:\n" + str(e))
		startButton['state'] = 'normal'
		stopButton['state'] = 'disabled'
		return
	test_file_name += '.csv' #добавляем расширение
	global stop
	stop = 0

	t1 = Thread(target=packet_capture, args=(test_file_name,))
	t1.start()
	label['text'] = "Захват пакетов..."

#завершение выполнения захвата
def stop_capture():
	stopButton['state'] = 'disabled'
	global stop
	stop = 1
	label['text'] = 'Пожалуйста, подождите'
	root.destroy()

#главное окно
root = tk.Tk()
root.title("Захват пакетов")
root.geometry("450x100")
root.resizable(False,False)
root.option_add('*Dialog.msg.font', 'Helvetica 12')

app = tk.Frame(root)
app.grid()

#создаем кнопки
startButton = tk.Button(app, text="Новый захват", width=15, font=('Courier',12), bg='gray', command=lambda: start_capture())
stopButton = tk.Button(app, text="Остановить захват", width=15, font=('Courier',12), bg='gray', state='disabled', command=stop_capture)

#вывод подсказок
CreateToolTip(startButton, text="Запустить новый захват данных")
CreateToolTip(stopButton, text="Остановить процесс захвата данных")
startButton.grid(row=0,column=0)
stopButton.grid(row=0,column=1)

label = tk.Label(app, text="Для лучшей точности проводите захват в течение 1-2 минут", font=('Courier',9), anchor='w', justify='left')
label.grid(row=1, column=0, columnspan=2)

app.mainloop()
