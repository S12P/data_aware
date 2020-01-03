import os
import sys
from collections import defaultdict
from random import randint

dic = defaultdict(lambda : 0)

film = []
while True:
    if len(film) == 100:
        break
    b = randint(1, 17770)
    if  b not in film:
        film += [b]


for k in film:
    file = open("/Users/stephane/Desktop/download/training_set/mv_{:07}.txt".format(k), "r")
    #print("mv_{:07}.txt".format(k))
    file.readline()
    while True:
        try:
            t = file.readline()
            a,b,c = t.split(",")
            dic[a] += 1

        except:
            break
    file.close()
    
#film = list(map(str, film))
users = list(dict(sorted(dic.items(), key=lambda item: -item[1])).keys())[:100]

#print(users)
#print()
#print(film)

final = open("matrix.txt", "w")
#print(film)
for k in film:
    file = open("/Users/stephane/Desktop/download/training_set/mv_{:07}.txt".format(k), "r")
    file.readline()
    while True:
        try:
            t = file.readline()
            a,b,c = t.split(",")
            if str(a) in users:
                final.write(str(k) + "," + str(a) + "," + str(b) + "\n")

        except:
            break
    file.close()
final.close()

