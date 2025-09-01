import csv

while True:
    insee = input('city:')
    with open('resources/communes-france-datagouv-2025.csv', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1] == insee:  # 2ème colonne = code postal
                print(row)
