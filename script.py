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

dic_user = {}

for k in range(len(users)):
    dic_user[users[k]] = k

#print(dic_user, len(dic_user))
#print()
#print(film)

final = open("matrix.txt", "w")
#print(film)
count = 0
for k in film:
    file = open("/Users/stephane/Desktop/download/training_set/mv_{:07}.txt".format(k), "r")
    file.readline()
    while True:
        try:
            t = file.readline()
            a,b,c = t.split(",")
            if str(a) in users:
                final.write(str(count) + "," + str(dic_user[a]) + "," + str(b) + "\n") # film, user, note

        except:
            break
    count += 1
    file.close()
final.close()

