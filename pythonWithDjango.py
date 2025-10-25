# print("Hello world")
# name = input("What is your name: ")
# color = input("What is your favoratie color : ")
# print(f"My name is {name}, and my favorite colro is {color}")

# now converting lbs into kg

# w = input("enter the weight ")
# kg = int(w)*0.45
# print(kg)

# weigth = input("ENter the weight ")
# kg = int(weigth)*0.45
# print(kg)

# name = "Kamal hai bro tum bi na"
# print(name[0])
# print(name[-1])
# print(len(name))
# print(name.upper())
# print(name.lower())
# print(name.find('P'))
# print(name.replace('Kamal', 'Harmai'))
# print('Kamal' in name)
# x = 2.9
# print(round(x))
# print(abs(-2.9))
# import math

# print(math.ceil(2.9))
# print(math.floor(2.9))
    

# is_adult = True
# if (is_adult):
#     print("you can vote babay")
# else:

#     print("you can not")


# is_adult = False
# if (is_adult):
#     print("you can vote babay")
# else:
#     print("you can not")









# is_l = True
# is_h = False

# if is_l:
#     print("love is good ")
#     print("find someone")
# elif is_h:
#     print("it is hated world")
#     print("i hate her")
# else:
#     print("i do not wanna love it is fucking shit")















# price = 1000000
# is_crdiet = True
# if is_crdiet:
#     downpayment = 0.1*price
# else:
#     downpayment = 0.2*price
# print(downpayment)




# has_pwoer = True
# has_exp = True
# if has_pwoer and has_exp:
#     print("amzing programmer ")

# has_pwoer = True
# has_exp = False
# has_pwoer = False
# has_exp = False
# if has_pwoer or has_exp:
#     print("amzing programmer ")
# else:
#     print("nothgin ur ")


# number = 50
# if number<30:
#     print("less")
# else:
#     print("it is pka")




# name = "kamal"
# if len(name)<60:
#     print("enter the long name ")
# else:
#     print("otherwise you can go baby")



# wieght = int(input("enter the weirgh "))
# unit  = input(" (L) lbs or (K) kilos ")
# if unit.upper=='L':
#     converted = wieght*0.45
#     print(converted)
# else:
#     converted = wieght/0.45
#     print(converted)



# love = int(input("Enter the amount of your love? "))
# amunt = input(" (L) love or (H) hate ")
# if amunt.upper=='H':
#     let = love-20
#     print(f"I can not belive this amoutn of {let}")
# else:
#     let = love/20
#     print(f"wow amazing man {let}")




# i = 1
# while(i<5):
#     print("*" * i)
#     i = i+1
# print("Done")


# command = ""
# while command!="quit":
#     command = input("> ").lower()
#     if command == "start":
#         print("started")
#     elif command=="stop":
#         print("stoped")
#     elif command =="help":
#         print ("""
# start to start
# stip to stop
# help to help

# """)
# else:
#     print("i do not know sorry")




# prices  = [1,2,3]
# total = 0
# for price in prices:
#     total +=price
# print(total)


# for x in range(5):
#     for y in range(3):
#         print(x,y)



# nu = [2,3]
# for x in nu:
#     print("x" * x)


# l = [1,2,3,4,5]
# max = l[0]
# for i in l:
#     if i > max:
#         max = i
# print(max)


# matrix = [
#     [1,2,3],
#     [4,5,6],
#     [7,8,9]

# ]
# matrix[0][1] = 20
# print(matrix[0][1])
# for i in matrix:
#     for j in i:
#         print(j)



# numbers = [5,2,4,5]
# unique = []
# for number in numbers:
#     if number not in unique:
#         unique.append(number)
# print(unique)
# numbers.insert(0,4)
# print(numbers)
# # print(numbers.index[90])
# print(50 in numbers)
# print(4 in numbers)
# numbers.sort()
# print(numbers)



# l = [1,2,3,4]
# m = []
# for i in l:
#     if i not in m:
#         m.append(i)
# print(m)

# no = (1,2,3)
# # print(no[1])
# a,b,c = no
# print(b)


# phone = input("Phone")
# dm = {
#     "1":"one",
#     "2":"two",
#     "3":"three",
#     "4":"four"
# }
# output = ""
# for char in phone:
#     output+=dm.get(char, " ! ")
# print(output)



# message = input(">")
# kalimas = message.split(' ')
# # print(kalimas)
# emo = {
#     ":)": "😐",
#     "(:": "😐"
# }
# for i in message:
#     emo.get(i, message)
# print(emo)
# print(i)

# print(message.split(' '))



# class Point:
#     def __init__(self,x,y):
#         self.x = x
#         self.y = y
#     def move(self):
#         print("move")

#     def draw(self):
#         print("draw")
# p1 = Point(9,10)
# # print(Point)
# # print(p1)
# print(p1.x)
# print(p1.y)
# p1.move()
# p1.draw()
# p1.x = 90
# p1.y = 10
# print(p1.x)
# print(p1.y)
# print(p1.x + p1.y)






# class person:
#     def __init__(self, name):
#         self.name = name
#     def talk(self):
#         # print("he is talking")
#         print(f"Hey i am {self.name} wow!  ")
# obj = person(name = "Safiullah")
# # print(obj.name)
# obj.talk()


# class Mammal:
#     def walk(self):
#         print("walk")


# class Dog(Mammal):
#     pass

# class Cat(Mammal):
#     # pass
#     def annoying(self):
#         print("shut the fuck up")


# obj = Dog()
# obj.walk()
# obj1 = Cat()
# obj1.walk()
# obj1.annoying()

# class Cat:
#     def walk(self):
#         print("walk")






# import random
# for i in range(3):
#     print(random.randint(3,100))


# import random
# nu = ["jhon", "kamal", "kan", "sada"]
# print(random.choice(nu))
# print(nu)


# import random

# class ludo:
#     def roll(self):
#         first = random.randint(1,6)
#         second = random.randint(1,6)
#         return first, second
    
# obj = ludo()
# print(obj.roll())
