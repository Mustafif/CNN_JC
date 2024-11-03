import csv

def historical():
    data = csv.reader(open('Historical.csv', 'r'))
    # skip first row
    next(data)
    S = []
    for row in data:
        S.append(float(row[0].strip()))
    return S
