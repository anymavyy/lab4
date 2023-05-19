#15. Формируется матрица F следующим образом: скопировать в нее А и если в Е количество чисел,
# больших К в четных столбцах больше, чем сумма чисел в нечетных строках,
# то поменять местами С и Е симметрично, иначе В и С поменять местами несимметрично.
# При этом матрица А не меняется. После чего если определитель матрицы А больше суммы диагональных элементов матрицы F,
# то вычисляется выражение: A*AT – K * FТ, иначе вычисляется выражение (AТ +G-1-F-1)*K, где G-нижняя треугольная матрица,
# полученная из А. Выводятся по мере формирования А, F и все матричные операции последовательно.

import numpy as np
import matplotlib.pyplot as plt
print("Введите число N не меньше 6:", end='')
n = int(input())
print("Введите число K:", end='')
k = int(input())
K = k#K в массив для умножения потом
A = np.random.randint(-10,10, size=(n,n),dtype="int64")#создание основной матрицы
Amo = np.copy(A)#копирование основ. матрицы
F = np.copy(A)#копирование основ. матрицы
At = np.zeros((n,n))#созадние нулевой
Ft = np.zeros((n,n))#созадние нулевой
kol = 0#количество чисел, больших К в четных столбцах
SUMA = 1#сумма чисел в нечетных строках
suma = 0
print('Основная матрица')
print(F)
for i in range(n//2, n):
    for j in range(n//2,n):
        if j % 2 !=0 and F[i,j] > k:
            kol += 1
for i in range(n//2, n):
    for j in range(n//2,n):
        if i % 2 ==0:
            SUMA += F[i,j]
if kol > SUMA:
    print('Преобразованная матрица')
    for i in range(n // 2):
        for j in range(n // 2, n - 1):
            F[i, j], F[(n - 1) - i, j] = F[(n - 1) - i, j], F[i, j]
elif kol <= SUMA:
    print('Преобразованная матрица')
    for i in range(n // 2):
        for j in range(n // 2):
            F[i, j], F[(n - 1) - i, j] = F[(n - 1) - i, j], F[i, j]
print()
print(F)
op = int(np.linalg.det(A))#функция для поиска определителя
for i in range(0,n):#сумма диагональных элементов
    suma = suma + F[i,i]
print(' ')
for i in range(0,n):
    suma = suma + F[i, n - 1 - i]
print('Определитель A:', op)
print('Сумма диагональных чисел F:', suma)
print('')
if op > suma:
    At = np.transpose(A)#функция транспортирования
    Proizv = At * A
    print("A*A транспонированная")
    print(Proizv)
    print('')
    Ft = np.transpose(F)#функция транспортирования
    print("F транспонированная")
    print(Ft)
    print('')
    FtK = K * Ft
    print(" K *F транспонированная")
    print(FtK)
    print('')
    print(" A*A транспонированная - K *F транспонированная")
    print(Proizv - FtK)
elif op < suma:
    print("A транспонированная")
    At = np.transpose(A)#функция транспортирования
    print(At)
    print('')
    print('G -1')
    G1 = np.linalg.inv(np.copy(np.tril(A)))# tril - нижняя треугольная матрица, linalg.inv - обратная матрица
    print(G1)
    print('')
    Ft = np.transpose(F)
    print("F -1")
    F1 = np.linalg.inv(F)
    print(F1)
    print('')
    print("(At + G1 - F1)*K")
    ND = (At + G1 - F1)*K
    print(ND)
skal1 = [np.mean(abs(F[i, ::])) for i in range(n)]
skal1 = int(sum(skal1))
fixg, skal2 = plt.subplots(2, 2, figsize=(11, 8))
x = list(range(1, n + 1))
for j in range(n):
    y = list(F[j, ::])
    skal2[0, 0].plot(x, y, ',-', label=f"{j + 1} строка.")
    skal2[0, 0].set(title="График с использованием функции plot:", xlabel='Номер элемента в строке',
                    ylabel='Значение элемента')
    skal2[0, 0].grid()
    skal2[0, 1].bar(x, y, 0.4, label=f"{j + 1} строка.")
    skal2[0, 1].set(title="График с использованием функции bar:", xlabel='Номер элемента в строке',
                    ylabel='Значение элемента')
    if n <= 10:
        skal2[0, 1].legend(loc='lower right')
        skal2[0, 1].legend(loc='lower right')
exp = [0] * (n - 1)
exp.append(0.1)
sizes = [round(np.mean(abs(F[i, ::])) * 100 / skal1, 1) for i in range(n)]
skal2[1, 0].set_title("График с ипользованием функции pie:")
skal2[1, 0].pie(sizes, labels=list(range(1, n + 1)), explode=exp, autopct='%1.1f%%', shadow=True)
def map(data, row_labels, col_labels, grap3, bar_gh={}, **kwargs):
    da = grap3.imshow(data, **kwargs)
    bar = grap3.figure.colorbar(da, ax=grap3, **bar_gh)
    grap3.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    grap3.set_yticks(np.arange(data.shape[0]), labels=row_labels)
    return da, bar
def annoheat(da, data=None, textcolors=("black", "white"), threshold=0):
    if not isinstance(data, (list, np.ndarray)):
        data = da.get_array()
    gh = dict(horizontalalignment="center", verticalalignment="center")
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            gh.update(color=textcolors[int(data[i, j] > threshold)])
            text = da.axes.text(j, i, data[i, j], **gh)
            texts.append(text)
    return texts
da, bar = map(F, list(range(n)), list(range(n)), grap3=skal2[1, 1], cmap="magma_r")
texts = annoheat(da)
skal2[1, 1].set(title="Создание аннотированных тепловых карт:", xlabel="Номер столбца", ylabel="Номер строки")
plt.suptitle("Использование библиотеки matplotlib")
plt.tight_layout()
plt.show()


