import random
import numpy as np

def gen_matix():
    rows = int(input())
    cols = int(input())
    matr = list()

    print(f"{rows} {cols}")

    for _ in range(cols):
        row = list()
        for _ in range(rows):
            num = random.randint(1,5)
            print(num, end=" ")
            row.append(num)
        matr.append(row)
        #
        print()
    #
    return matr

def dump(matr):
    shape = matr.shape
    print(f"{shape[0]} {shape[1]}")
    for row_it in range(shape[0]):
        for col_it in range(shape[1]):
            print(matr[row_it][col_it], end=" ")
        #
        print()

def main():
    matr1 = gen_matix()
    print()
    matr2 = gen_matix()
    
    matr1 = np.array(matr1)
    matr2 = np.array(matr2)

    res = np.dot(matr1, matr2)

    print()
    dump(res)

if __name__ == '__main__':
    main()