# # print("Hello aram")

# # def ram(n):
# #     print(n)
# # ram("I love you")


# # print("Hello world")
# # numbers = [1,2,3,4,4,5,6,7,-7,-5,-4,-3]
# # positive_numbers = 0
# # for num in numbers:
# #     if num>0:
# #         positive_numbers+=1
# # print(positive_numbers)



# # 
# # n = 10
# # sum_even = 0
# # for i in range(1, n+1):
# #     if i%2==0:
# #         sum_even+=1
# # print(sum_even)


# # n = 10 
# # for i in range(1,11):
# #     print(n, " X ", i, " = ", n*i)

# # n = 10 
# # for i in range(1,11):
# #     if i==5:
# #         continue
# #     print(n, " X ", i, " = ", n*i)



# # input_str = "Python"
# # revers_str = ""

# # for char in input_str:
# #     # print(input_str)
# #     # revers_str = revers_str+char
# #     revers_str = char+revers_str
# # print(revers_str)



# # input_str = "teeter"
# # for char in input_str:
# #     # print(char)
# #     if input_str.count(char)==1:
# #         print(char)
# #         break



# # number  = 5
# # factorial = 1

# # while number>0:
# #     factorial  = factorial*number
# #     number = number-1

# # print(factorial)



# # while True:
# #     number = int(input("Enter the nuber? "))

# #     if 1<= number <=10:
# #         print("wow")
# #         break
# #     else:
# #         print("invalid ")

# # # 
# # number = 28
# # # number = 29
# # is_prime = True

# # if number>1:
# #     for i in range(2, number):
# #         if number%i==0:
# #             is_prime = False
# #             break
    

# # print(is_prime)


# # items = ["apple", "bannaa", "orange", "apple"]

# # unique_item = set()
# # for item in items:
# #     if item in unique_item:
# #         print(item)
# #         break
# #     unique_item.add(item)

















# # import time
# # aite_time = 1
# # iterate_time = 5
# # attempts = 0

# # while attempts<iterate_time:
# #     print(attempts +1, aite_time)
# #     time.sleep(aite_time)
# #     aite_time*=2
# #     attempts+=1


# # import time
# # f = open('open.py')
# # print(f.readlines())

# # for line in open('open.py'):
# #     print(line, end=' ')



# # for line in open('open.py'):
# #     print(line, end=' ')

# # f = open('open.py')
# # while True:
# #     line = f.readline()
# #     if not line: break
# #     print(line) 


# # f  = open('open.py')
# # while True:
# #     line = f.readline()
# #     if not line:break
# #     print(line, end=' ')

# # for line in open('open.py'):
#     # print(line)



# # import streamlit as st
# # import pandas as pd
# # import plotly.express as px
# # import numpy as np
# # import plotly.graph_objects as go

# # st.write("ProfessionalDashbord ")








# # ##importar as bibliotecas 
# # import streamlit as st 
# # import pandas as pd
# # import plotly.express as px 
# # import plotly.graph_objects as go

# # ##Abrir no csv
# # df = pd.read_csv(r"D:\hello\NDVI__Precipitation__and_Temperature_Data.csv")

# # ##Converter a primeira para
# # df['Date']= pd.to_datetime(df['Date'])

# # df['Year-Month']= df['Date'].dt.to_period('M')

# # ##Agregar dados 
# # df_monthly = df.groupby('Year-Month').agg({
# #     'Precipitation (mm)':'sum',
# #     'Temperature (°C)':'mean',
# #     'NDVI': 'mean'  
    
# # }).reset_index()

# # ##Gerar nosso
# # ndvi_trace = go.Scatter(x=df_monthly['Year-Month'].astype(str),
# #                         y=df_monthly['NDVI'],
# #                         mode='lines',
# #                         name='NDVI',
# #                         line=dict(color='green'))

# # precipitation_trace = go.Bar(x=df_monthly['Year-Month'].astype(str),
# #                         y=df_monthly['Precipitation (mm)'],
# #                         name='Precipitação',
# #                         yaxis='y2',
# #                         opacity=0.6,
# #                         marker=dict(color='blue'))


# # layout =go.Layout(
# #     title='NDVI e Precipitação Mensal',
# #     xaxis=dict(title='Mês'),
# #     yaxis=dict(title='NDVI', range=[0,1]),
# #     yaxis2=dict(title='Precipitação', overlaying='y', side='right'),
# #     legend=dict(x=0, y=1.1, orientation='h'),
# #     barmode='overlay'
    
# # )

# # fig1 = go.Figure(data=[ndvi_trace,precipitation_trace], layout=layout)

# # ###Definir a figura 2 
# # # Gráfico de heatmap para Temperatura
# # heatmap_data = df_monthly.pivot_table(index=df_monthly['Year-Month'].dt.year, columns=df_monthly['Year-Month'].dt.month, values='Temperature (°C)')

# # fig2= px.imshow(
# #     heatmap_data,
# #     labels=dict(x= 'Month',y ='Year', color='Temperature (°C)'),
# #     x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
# #     title="Temperature Heatmap Over 24 Months",
# #     color_continuous_scale="RdYlGn"
# # )

# # ##Set configuração da pagina
# # st.set_page_config(layout='wide',
# #                    initial_sidebar_state='expanded')

# # st.sidebar.write('Pro App criado para apresentação dos resultados de NDVI, Precipitação e Temperatura da Fazenda Youtube')
# # st.sidebar.image(r'D:\hello\ndvi.png')

# # ##Criar um titulo
# # st.title('PRO, Precipitação e Temperatura Dashboard')

# # st.dataframe(df_monthly, width=1200, height=400)

# # ##Colunas 
# # col1, col2 = st.columns([0.5,0.5])

# # with col1:
# #     st.subheader('NDVI e Precipitação')
# #     st.plotly_chart(fig1)
    
# # with col2:
# #     st.subheader('Temperatura')
# #     st.plotly_chart(fig2)











# # print("Hello world")
# # cal = 24;
# # units = "hourse"

# # def day(no_days):
# #     if no_days>0:

# #         return f"{no_days} are {cal*no_days} {units}"
# #     else:
# #         print("you entered the negative numbers")
# # define = day(-10)
# # print(define)



# a  = 1
# b = '1'
# c = "1"
# d = 1.0
# print(a)
# print(b)
# print(c)
# print(d)
# print(type(a))
# print(type(b))
# print(type(c))
# print(type(d))
# x = "Python"
# y = 'OneShot'
# print(x)
# print(y)
# print(id(a))
# python = 1
# py_thon = 1
# _python = 1
# Python = 1
# PYTHON = 1
# python1 = 1
# # if = 1        # invalid resevred wrd
# # 1python = 1   # invalid vairabl ename
# # py$thon = 1
# # pyth@thon = 1    # invalid also

# variable1, variable2, vairalble3 = '1', '2', '3'
# variable1 = variable2=vairalble3=5
# print(variable1)
# print(variable2)
# print(vairalble3)

# # comments:
# '''
# This is an example fo the comment multiline ones
# '''
# print('commnets')

# # dtypes:
# d1 = 5
# d2 = "safiullah"
# d3 = 5.0
# d4 = True
# d5 = 5j
# d6 = [1,2,3,4,5]
# d7 = {1,2,3,9}
# d8 = (91,2,4,5)
# d9 = {"naam": 'love', 'kamal': 'jamal'}
# d10 = None
# print(type(d1))
# print(type(d2))
# print(type(d3))
# print(type(d4))
# print(type(d5))
# print(type(d6))
# print(type(d7))
# print(type(d8))
# print(type(d9))
# print(type(d10))

# A = '''Hello and welcome dosto 
# to 5 minutese engineering,
# aaj ka video bada kamal ka hone wala hai'''
# print(A)


# B = """Hello and welcome dosto 
# to 5 minutese engineering,
# aaj ka video bada kamal ka hone wala hai"""
# print(B)

# C = 'SafiullahNasri'
# print(C[0])
# print(C[1])
# print(C[2])
# print(C[4])
# print(C[5])
# print(C[6])
# print(C[7])
# print(C[8])
# print(C[9])
# print(C[10])

# print(len(A))
# print(len(B))
# print(len(C))



# # type conversion
# intt = 5
# flt = 5.0
# add = intt+flt
# print(type(add))
# add


# # intt = 5
# # strr = '5'
# # adddd = intt+strr
# # print(type(intt+strr))
# # adddd


# intt = 5
# strr = '5'
# adddd = intt+int(strr)
# print(type(adddd))
# adddd





# # operators in the python

# x = 2
# y = 5
# print(x+y)
# print(x-y)
# print(x*y)
# print(x/y)
# print(x%y)
# print(x**y)


# # Comparison operators

# print(x==y)
# print(x!=y)
# print(x<y)
# print(x>y)
# print(x<=y)
# print(x>=y)
# # print(x==y)

# # Assignments operators


# k = 10
# print(k)
# k+=10
# print(k)
# k-=10
# print(k)
# k*=10
# print(k)
# k/=10
# print(k)
# k**=10
# print(k)




# m = 1
# n = 2
# print(m<2 and n>1)
# print(m>2 and n>1)
# print(m<2 and n<1)
# print(m>2 and n<1)

# print(m<2 or n>1)
# print(m>2 or n>1)
# print(m<2 or n<1)
# print(m>2 or n<1)


# print(not(m<2 and n>1))
# print(not(m>2 and n>1))
# print(not(m<2 and n<1))
# print(not(m>2 and n<1))

# print(True and True)
# print(True and False)
# print(False and True)
# print(False and False)

# print(True or True)
# print(True or False)
# print(False or True)
# print(False or False)


# u = 2
# v = 1
# print(u & v)
# print(u | v)
# print(u ^ v)
# print(u >> 2)
# print(u << 2)


# # # now the input adn output faunda

# # # num = input("Enter a number? ")
# # print("You entered :", num)
# # print(type(num))
# # # num = int(input("Enter the number? "))
# # print(type(num))
# # print(num)


# print("Professional", end= ' ')
# print("Hello")
# print("kamal", 5, "jamal", sep=',')
# print("kamal" + "jamal")


# num1 = 4
# num2 = 3
# print("The expressed number is {}, and {}".format(num1, num2))


# # if else now 
# var1 = 10
# var2 = 20
# if var1 !=var2:
#     print("Not equal")

# if var1==var2:
#     print("not equal")

# if var1<var2 & var2>5:
#     print("if block") 

# if var1>var2 or var1<var2:
#     print("no block")

# if not var1==var2:
#     print("Kmala hai main")

# if var1!=var2:
#     print("True")
# elif var1==var2:
#     print("False")

# # for loop in the python


# abc = (1,2,3,4,5,6,7,8,9,10)
# for i in abc:
#     print(i)


# for i in abc:
#     if i==5:
#         break
#     print(i)

# for i in abc:
#     if i==5:
#         continue
#     print(i)


# for i in "safiullah":
#     print(i)
# name = "professinal"
# for i in name:
#     print(i)
# print(name)


# for i in range(1,10):
#     print(i, end=" ")

# for i in range(5,10):
#     print(i)


# for i in range(1,10,2):
# # here to is the difference it mena after printing every numver there must be the diffeernce between the numbers
#     print(i)


# for i in "kamal":
#     print(i)
#     for j in "professional":
#         print(j)


# # while loop in the python

# x = 1
# while(x<10):
#     print(x)
#     x +=1

# i = 1
# while(i<10):
#     print(i)
#     i+=2


# j = 1
# while j<=10:
#     if j==5:
#         break
#     j+=1
#     print(j)




# # x = 1
# # while x<5:
# #     y = 1
# #     while y<5:
# #         print(x+y)
# #         y+=1
# #     print("\n")
# #     x+=1



# # Now we talk about the list in the python

# l1 = [1,2,3,4,5]
# l2 = ["one", "two", "three"]
# l3 = [True, False]
# l4 = [1.1,2.2,3.3,4.4,5.5]
# print(type(l1))
# print(type(l2))
# print(type(l3))
# print(type(l4))
# print(l1[0])
# print(l1[1])
# print(l1[2])
# print(l1[3])
# print(l1[4])
# print(l2[0])
# print(l2[1])
# print(l2[2])
# print(l3[0])
# print(l3[1])
# l1[4] = 6
# print(l1[4])
# # l1.appen(7)
# # print(l1)
# l1.insert(4,3)
# print(1)
# l2.extend(l1)
# print(l2)
# l1.remove(6)
# print(l1)
# l1.pop()
# print(l1)
# l1.sort(reverse=True)
# print(l1)
# l5 =l1.copy()
# print(l5)
# l1.clear()
# print(l1)
# l1.append(5)
# l1.append(5)
# l1.append(5)
# l1.append(5)
# l1.append(5)
# print(l1)
# # del l1()
# # print(l1)
# tup  = (1,2,3,4)
# tuplist = list(tup)
# tuplist.append(5)
# tup = tuple(tuplist)
# print(tup)

# s = {1,1,2,3,4,5,6}
# s1 = {True, False, 4,5,6,8}
# print(1 in s)
# print(10 in s)
# print(s)
# s.add(100)
# print(s)
# s.remove(4)
# print(s)
# s_new = s.union(s1)
# print(s_new)



# # now we talk about the dictionaries
# d = {
#     "name": "ramzia",
#     "rollno": 12
# }
# print(d)


# d2 = dict(id=1, name = "kamal", age=90)
# print(d2)
# print(d.keys())
# print(d2.values())


# # Function in the python programming language
# def function1():
#     print("Hello world! ")
# function1()


# def function2(plus):
#     print("Hello world! ", plus)
# function2("to 5me")


# def name(fname, lname):
#     print(fname + lname)
# name("kamal", "khan")



# def dic(**students):
#     print(students['s1'], students['s2'], students['s3'])
# dic(s1 = "kamal", s2 = 'jan', s3 = 'khan')

# def sum(a,b):
#     return a+b 
# print(sum(9,4))


# # Recursion after fucntions
# def norm(x):
#     while x>0:
#         print(x)
#         x-=1
# norm(10)

# def rf(a):
#     print(a)
#     if (a>0):
#         rf(a-1)
# rf(10)


# def fac(n):
#     if n==1:
#         return 1
#     else:
#         return n*fac (n-1)
# print(fac(5))






# import csv
# with open('D:\hello\diabetes.csv', 'r') as file:
#     reader = csv.reader(file)
#     for i in reader:
#         print(i)


# import csv
# with open("pro.csv", 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow([1,2,3])
#     writer.writerow([56,45,34])

# import csv
# with open('pro.csv', 'r') as file:
#     reader = csv.reader(file)
#     for i in reader:
#         print(i)


# exception handling:

# try:
#     lsit = [1,2,3,4,5]
#     print(list[9999])
# except IndexError:
#     print("indexing error")


# def name(x):
#     print(a)
# try:
#     name(5)
# except ZeroDivisionError():
#     print("ksjflsfk")
# except NameError():
#     print("sjfkldlkj")


# class CustomException(Exception):
#     print("wow this is the custom exceptin")
#     pass
# threshold = 35
# try:
#     marks = int(input("enter thr marks"))
#     if marks<threshold:
#         raise CustomException
#     else:
#         print("pass")

# except CustomException:
#     print("fial")
    




# class and objects:

# class class1:
#     d1 = 'kaihan'
#     d2 = 2
# obj = class1()
# print(obj.d1)
# print(obj.d2)

# obj1 = class1()
# print(obj1.d1)
# obj1.d1 = "kamlhain"
# print(obj1.d1)



# slef key word now


# class c1():
#     name = "kaml"
#     rool_no = 4

#     def fc(self):
#         print("Name is: ", self.name)
# obj = c1()
# obj. fc()
# obj.name =  "kamalJan"
# # obj.rool_no()
# obj.fc()




# class c1():
#     def __init__(self, name):
#         self.name = name
#     ro_no = 1
    
#     def abc(self):
#         print("Name is : ", self.name)
# obj = c1("Safiullah")
# print(obj.name)
# obj.abc()


# class parent():
#     name = ""
#     def m1(self):
#         print("Parent in the class")
# class child(parent):

#     def m2(self):
#         print("name is: ", self.name )
# obj = child()
# obj.name = "Safiullah"
# obj.m1()
# obj.m2()






# class c():
#     x = 1
#     def __init__(self):
#         self.x = 5
#         print(self.x)
# obj = c()
# print(obj.x)




# class parent():
#     x = 1
# class child(parent):
#     def __init__(self):
#         print(self.x)
# obj = child()
# print(obj.x)



# class A():
#     def f1(self):
#         print("It is A")
# class B():
#     def f2(self):
#         print("It is B")
# class C(A, B):
#     def f3(self):
#         print("It is C")
# obj = C()
# obj.f1()
# obj.f2()
# obj.f3()





# fun = lambda: print("Getting Python")
# fun()



# l = lambda x: print(x)
# l("kamlhai to yar")

# con = lambda x,y : x if x>y else y
# # con(5,6)
# print(con(4,5))



# listttt = [1,2,3,4,5]
# print(dir(listttt))
# print(iter(listttt))


# listttt = [1,2,3,4,5]
# ite = iter(listttt)
# print(next(ite))
# print(next(ite))





# class iterator():
#     def __init__(self, val):
#         self.val = val

#     def __iter__(self):
#         self.num = 1
#         return self
    
#     def __next__(self):
#         if self.num<=self.val:
#             final = self.num*5
#             self.num+=1
#             return final 
#         else:
#             raise StopIteration
# obj = iterator()
# itr = iter(obj) 
# print(next(itr))      
# print(next(itr))      
# print(next(itr))      





# def generator():
#     n = 0

#     n = n+5
#     yield n
#     n = n+5

#     yield n
#     n = n+5

#     yield n
#     n = n+5
# obj = generator()
# print(obj)
# print(next(obj))
# print(next(obj))
# print(next(obj))
# print(next(obj))



# def gne(n):
#     x = 5
#     while x<n:
#         yield x
#         x = x+5

# for i in gne(100):
#     print(i)



# def a ():
#     def b():
#         print(a)
#     b()
# a(3)



# def abc():
#     def xyz():
#         print(abc)
#     return xyz
# # x = abc(5)
# # x()


# def a(a):
#     def b(b):
#         return a*b
#     return b()
# x = a(5)
# y = a(5)
# print(x(4))
# print(y(3))



# def decorated_fun(fun):
#     def inner():
#         fun()
#         print("Dosto")
#     return inner
# def origional():
#     print("hello ji", end=" ")
# d_f = decorated_fun(origional)
# d_f()



# def decorated_fun(fun):
#     def inner():
#         fun()
#         print("Dosto")
#     return inner
# @decorated_fun
# def origional():
#     print("hello ji", end=" ")
# # d_f = decorated_fun(origional)
# # d_f()
# origional()




# ticket = 100

# while True:
#     n = input("uy tickets")
#     n = int(n)
#     ticket -=n
#     print(f"tickets left: {ticket}")

# Febonic numbers:

# a = 0 
# b = 1

# for i in range(10):
#     c = a+b
#     a = b
#     b = c
#     print(c)


# a = 0
# b = 1

# for i in range(10):
#     print(a)

#     c = a+b
#     a = b
#     b = c



# def say_hello():
#     print("hello world")
       
# say_hello()



# # name = "kaiha"de
# # print(f"Hello { name} ")


# def add(a,b):
#     total = a+b
#     print("{} + {} = {} ".format(a,b,total))
#     return total

# x = add(4,5)
# print(x)



# for i in dir(list):
#     print(i)

def fac(n):
    if n==0:
        return 1
    # print(n)
    return n*fac(n-1)
print(fac(5))