import numpy as np
import sys


def read_file(filename):
    try:
        f = open(filename, 'r')
        lines = f.readlines()
        f.close()

        clean_lines = []
        for line in lines:
            if line.strip() != '':
                clean_lines.append(line.strip())

        if len(clean_lines) == 0:
            print("Ошибка: файл пуст")
            return None, None, None

        first_line = clean_lines[0].split()
        n = None
        for item in first_line:
            try:
                item = item.replace(',', '.')
                n = int(float(item))
                break
            except:
                continue

        if n is None:
            print("Ошибка: не удалось определить размерность")
            return None, None, None

        if n > 20:
            print("Ошибка: размерность превышает 20")
            return None, None, None

        if len(clean_lines) < n + 1:
            print("Ошибка: недостаточно строк в файле")
            return None, None, None

        A = []
        B = []

        for i in range(n):
            row_data = clean_lines[i + 1].split()
            values = []
            for x in row_data:
                try:
                    x = x.replace(',', '.')
                    values.append(float(x))
                except:
                    continue

            if len(values) < n + 1:
                print(f"Ошибка в строке {i + 2}")
                return None, None, None

            A.append(values[:n])
            B.append(values[n])

        return A, B, n

    except FileNotFoundError:
        print("Ошибка: файл не найден")
        return None, None, None
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None, None, None


def input_keyboard():
    while True:
        try:
            n = int(input("Введите размерность матрицы n (n <= 20): "))
            if 1 <= n <= 20:
                break
            else:
                print("Размерность должна быть от 1 до 20")
        except ValueError:
            print("Ошибка: введите целое число")

    A = []
    print("Введите коэффициенты матрицы A построчно (через пробел):")
    for i in range(n):
        while True:
            try:
                row = input(f"Строка {i + 1}: ").strip().replace(',', '.')
                vals = list(map(float, row.split()))
                if len(vals) != n:
                    print(f"Ошибка: нужно ввести {n} чисел")
                    continue
                A.append(vals)
                break
            except ValueError:
                print("Ошибка: введите числа, разделенные пробелами")

    print("Введите вектор правых частей B (через пробел):")
    while True:
        try:
            b = input().strip().replace(',', '.')
            B = list(map(float, b.split()))
            if len(B) != n:
                print(f"Ошибка: нужно ввести {n} чисел")
                continue
            break
        except ValueError:
            print("Ошибка: введите числа")

    return A, B, n


def gauss_pivot(A, B):
    n = len(A)

    AA = [row[:] for row in A]
    BB = B.copy()

    M = []
    for i in range(n):
        M.append(AA[i] + [BB[i]])

    swaps = 0

    for k in range(n):
        max_elem = abs(M[k][k])
        max_row = k
        for i in range(k + 1, n):
            if abs(M[i][k]) > max_elem:
                max_elem = abs(M[i][k])
                max_row = i

        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]
            swaps += 1

        if abs(M[k][k]) < 1e-15:
            raise ValueError("Матрица вырождена или близка к вырождению")

        for i in range(k + 1, n):
            if M[i][k] != 0:
                coef = M[i][k] / M[k][k]
                for j in range(k, n + 1):
                    M[i][j] -= coef * M[k][j]

    det = 1
    for i in range(n):
        det *= M[i][i]
    if swaps % 2 == 1:
        det = -det

    x = [0] * n
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s += M[i][j] * x[j]
        x[i] = (M[i][n] - s) / M[i][i]

    return x, det, M, swaps


def calculate_residuals(A, B, x):
    n = len(A)
    r = []
    for i in range(n):
        s = 0
        for j in range(n):
            s += A[i][j] * x[j]
        r.append(s - B[i])
    return r


def main():
    print("ЛАБОРАТОРНАЯ РАБОТА №1")
    print("Решение СЛАУ методом Гаусса с выбором главного элемента по столбцу")
    print("Вариант 13")

    while True:
        print("\nВыберите способ ввода данных:")
        print("1 - с клавиатуры")
        print("2 - из файла")
        print("0 - завершение работы")

        choice = input("> ")

        if choice == '0':
            print("\nЗавершение работы.")
            break

        A = None
        B = None
        n = 0

        if choice == '1':
            A, B, n = input_keyboard()
        elif choice == '2':
            fname = input("Введите имя файла: ")
            A, B, n = read_file(fname)
        else:
            print("Ошибка: неверный выбор")
            continue

        if A is None:
            print("Не удалось загрузить данные")
            continue

        try:
            x, det, triang_matrix, swaps = gauss_pivot(A, B)

            print("\nТреугольная матрица (расширенная):")
            for i in range(n):
                line = ""
                for j in range(n):
                    line += f"{triang_matrix[i][j]:12.6f} "
                line += f"| {triang_matrix[i][n]:12.6f}"
                print(line)

            print("\nВектор неизвестных:")
            for i in range(n):
                print(f"x{i + 1} = {x[i]:.10f}")

            residuals = calculate_residuals(A, B, x)
            print("\nВектор невязки:")
            for i in range(n):
                print(f"r{i + 1} = {residuals[i]:.2e}")

            print("\nСРАВНЕНИЕ С БИБЛИОТЕЧНЫМ РЕШЕНИЕМ (NumPy)")

            A_np = np.array(A, dtype=float)
            B_np = np.array(B, dtype=float)
            x_numpy = np.linalg.solve(A_np, B_np)
            det_numpy = np.linalg.det(A_np)

            print(f"\nОпределитель (метод Гаусса): {det:.10f}")
            print(f"Определитель (NumPy): {det_numpy:.10f}")

            print("\nВектор неизвестных X (NumPy):")
            for i in range(n):
                print(f"x{i + 1} = {x_numpy[i]:.10f}")

            print("\nСравнение решений (разница |x_гаусс - x_numpy|):")
            max_diff = 0
            for i in range(n):
                diff = abs(x[i] - x_numpy[i])
                print(f"Δx{i + 1} = {diff:.2e}")
                if diff > max_diff:
                    max_diff = diff
            print(f"Максимальная разница: {max_diff:.2e}")

            print("\nАнализ результатов:")
            if max_diff < 1e-10:
                print("Решения практически совпадают с высокой точностью.")
                print("Это свидетельствует о корректной реализации метода Гаусса")
                print("и хорошей обусловленности матрицы.")
            elif max_diff < 1e-6:
                print("Наблюдаются небольшие расхождения в решениях.")
                print("Это может быть связано с погрешностями округления.")
            else:
                print("Обнаружены значительные расхождения в решениях.")
                print("Возможные причины: плохая обусловленность матрицы,")
                print("накопление погрешностей округления или близость матрицы к вырожденной.")

        except Exception as e:
            print(f"\nОшибка при решении: {e}")


if __name__ == "__main__":
    main()