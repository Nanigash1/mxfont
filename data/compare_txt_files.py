import os
kz = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "А", "Б", "В", "Г", "Д", "Е", "Ж", "З", "И", "Й", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Э", "Ю", "Я", "а", "б", "в", "г",
"д", "е", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я", "Ё", "ё", "Ә", "ә", "Ғ", "ғ", "Қ", "қ", "Ң", "ң", "Ө", "ө", "Ұ", "ұ", "Ү", "ү", "Һ", "һ", "І", "і"]
def compare_txt_files(folder1, folder2):
    # Проверка существования папок
    if not os.path.exists(folder1):
        print(f"Папка '{folder1}' не существует.")
        return
    if not os.path.exists(folder2):
        print(f"Папка '{folder2}' не существует.")
        return

    # Получение списка всех файлов в папках
    files1 = [f for f in os.listdir(folder1) if f.endswith(".txt")]
    files2 = [f for f in os.listdir(folder2) if f.endswith(".txt")]

    # Пересечение имен файлов для сравнения
    common_files = set(files1) & set(files2)

    for file_name in common_files:
        file1_path = os.path.join(folder1, file_name)
        file2_path = os.path.join(folder2, file_name)

        with open(file1_path, 'r', encoding='utf-8') as file1, open(file2_path, 'r', encoding='utf-8') as file2:
            file1_content = file1.read()
            file2_content = file2.read()

            set1 = set(file1_content)
            set2 = set(file2_content)

            differences = set1.symmetric_difference(set2)
            if differences:
                # print(f"Различия в файле: {file_name}")
                for diff in differences:
                    if diff in set1 and diff not in set2 and diff in kz:
                        print(f"Символ '{diff}' присутствует в {file1_path} и отсутствует в {file2_path}")
                    elif diff in set2 and diff not in set1 and diff in kz:
                        print(f"Символ '{diff}' присутствует в {file2_path} и отсутствует в {file1_path}")

# Пример использования функции
folder1 = r"C:\Users\Lenovo\Documents\GitHub\mxfontG\mxfont\data\ttfs\val"
folder2 = r"C:\Users\Lenovo\Documents\GitHub\content\mxfont\data\ttfs\val"
compare_txt_files(folder1, folder2)


