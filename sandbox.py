# Testing and playground
import numpy as np

a = [1, 2, 3]
b = [4, 5, 6]

for epoki in range(3):
    for lista1, lista2 in zip(a, b):
        #print(f"Iteracja nr {epoki}")
        print(lista1)
        print("---------")
        print(lista2)
        print("KONIEC FORA NR 2 NASTEPNA ITERACJA")