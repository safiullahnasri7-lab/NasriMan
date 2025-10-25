# from calculator import square


# def main







# print("Hello world")

# print("Hello world")

# email = input("What is your eamil?").strip()

# if "@" in email:
#     print("valid")
# else:
#     print("Invalid..")






# import re
# email = input("Enter your email here? ")

# if re.search("@", email):
#     print("valid")

# else:
#     print('invalid')





# name = input("Enter your name ").strip()
# if "," in name:
#     last, first = name.split(", ")

#     name = f"{first} {last}"
# print(f"hello, {name}")




# print("Hello world")


# for i in range(5):
#     print("Day")



# print("Hello world")
# # kamal hai yar
# print(9+1)


# a = ("kamla", "jamal", "khan")
# print(a)
# a = list(a)
# print(a)
# a.append("kamal")
# print(a)
# a = tuple(a)
# print(a)
# # a.append("kamal")


# a = ("kamal", "jamal", "khan", "khan", "khan")
# print(a.count("khan"))


# import json
# # solving some probelmsin the python realted to the list tupels and dictonaries int eh programming
# studentData = {"name": "kamal", "age": 21, "makrs": 99}
# # print(studentData)

# # data = json.dumps(studentData, indent=4, separators=(",", "="))
# # print(data)

# f = open("demo.json", "w")
# data = json.dumps(studentData,indent=4, sort_keys= True)
# f.write(data)
# print("data added to the json file...")



# ed = {"name":"Dewangak", "age": 21, "gender":"male"}
# print(ed)
# print(ed['age'])
# print(ed['gender'])
# print(ed['name'])
# print(ed.keys())
# print(ed.values())
# for i in ed:
#     print(i)
# for j in ed:
#     print(ed[i])


# for x,y in ed.items():
#     print(x,y)

# fucniton fo the dictionaries

# print(ed.get("name"))
# print(ed.items())
# print(ed.keys())
# print(ed.values())
# coppied = ed.copy()
# print(coppied)
#

# fucniton part two in the python of the dictionries
# print(ed.setdefault("age", 78))


# nested dicitonaries in the python



# e_dic = {1: {"Name": "kamal", "Age": 45, "Gender": "Male"},
#          2: {"Name": "lisa", "Age": 23, "Gender": "Female"},
#          3: {"Name": "Dewanagak", "Age": 67, "Gender":"Female"} }

# print(e_dic)
# print(e_dic[1]['Gender'])
# # 

# now the time of the problem solving 


# a = {"a": 12, "b": 34, "c":23,"d":87, "e":56}
# print(a.values())
# a = sorted(a.values())
# print(a)
# a = sorted(a.keys())
# print(a)
# print(a.keys())



# b = {}
# for i in range(1,16):
#     b[i] = i**2
# print(b)

# a = {}
# for i in range(1,10):
#     a[i] = i*i
# print(a)


# a = {"a": 1, "b": 2, "c":3,"d":4, "e":5}
# for i in a:
#     print(i)
#     for j in a:
#         print(a[j])
# for i in a:
#     print(a.values())
# a = {"a": 1, "b": 2, "c":3,"d":4, "e":5}
# mulResult = 1
# for i in a:
#     mulResult*=a[i]
# print(mulResult)

# pro =1
# for i in a:
#     pro*=a[i]
# print(pro)

# for i in a:
#     print(i)

# now we tlak about the sets in the python

# em = {}
# for i in range(1,10):
#     em[i] = i**2

# print(em)


# a = {"a":1, "b":2,"c":3}
# pro = 1
# for i in a:
#     pro*=a[i]
# print(pro)




# a = {}
# for i in range(1,10):
#     a[i] = i**2
# print(a)


# a = {"a":1, "b":2,"c":3}
# pro= 1
# for i in a:
#     pro*=a[i]
# print(pro)














# s = {1,2,3,4,6}
# print(max(s))
# print(min(s))

# a = {3,4,5,6}
# b = {39,48,5,6}
# c = {30,4,2,8}
# print(set(a) and set(b) and set(c))
# c = {30,4,2,8}
# c.discard(4)
# print(c)




print("Hello pyKaiahn Hakcers".split(" "))
print("Hello pyKaiahn Hakcers".split("PyKaiahn"))
print("Hello pyKaiahn Hakcers".split())



text = "KamaHai"
text = text.capitalize()
print(text)

print("Hello 1234".isalnum())
print("Hello".isalnum())
print("1234".isalnum())
print("Hello @1234".isalnum())
print("Hello @".isalnum())


my_list = [1,2]
my_list.append(3)
print(my_list)


l = [1,2,3]
n_l = l.copy()
print(n_l)


name = [1,2,3,44,4,4,4,4]
print(name.count(4))