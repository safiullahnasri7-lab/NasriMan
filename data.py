# from function import *
# from time import sleep

# for action in actions: 
#     for sequence in range(no_sequences):
#         try: 
#             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#         except:
#             pass

# # cap = cv2.VideoCapture(0)
# # Set mediapipe model 
# with mp_hands.Hands(
#     model_complexity=0,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as hands:
    
#     # NEW LOOP
#     # Loop through actions
#     for action in actions:
#         # Loop through sequences aka videos
#         for sequence in range(no_sequences):
#             # Loop through video length aka sequence length
#             for frame_num in range(sequence_length):

#                 # Read feed
#                 # ret, frame = cap.read()
#                 frame=cv2.imread('Image/{}/{}.png'.format(action,sequence))
#                 # frame=cv2.imread('{}{}.png'.format(action,sequence))
#                 # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#                 # Make detections
#                 image, results = mediapipe_detection(frame, hands)
# #                 print(results)

#                 # Draw landmarks
#                 draw_styled_landmarks(image, results)
                
#                 # NEW Apply wait logic
#                 if frame_num == 0:
#                     cv2.putText(image, 'STARTING COLLECTION', (120,200), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
#                     cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                     # Show to screen
#                     cv2.imshow('OpenCV Feed', image)
#                     cv2.waitKey(200)
#                 else: 
#                     cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                     # Show to screen
#                     cv2.imshow('OpenCV Feed', image)
                
#                 # NEW Export keypoints
#                 keypoints = extract_keypoints(results)
#                 npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
#                 np.save(npy_path, keypoints)

#                 # Break gracefully
#                 if cv2.waitKey(10) & 0xFF == ord('q'):
#                     break
                    
#     # cap.release()
#     cv2.destroyAllWindows()



# # from function import *   # This imports mediapipe_detection, draw_styled_landmarks, extract_keypoints
# # import cv2
# # import numpy as np
# # import os
# # from time import sleep
# # from ultralytics import YOLO

# # # Load YOLO model
# # model = YOLO("yolov8n-pose.pt")

# # # Create folders
# # for action in actions:
# #     for sequence in range(no_sequences):
# #         try:
# #             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
# #         except:
# #             pass

# # # Loop through actions
# # for action in actions:
# #     for sequence in range(no_sequences):
# #         for frame_num in range(sequence_length):

# #             # Read image (instead of webcam for this script)
# #             frame = cv2.imread('Image/{}/{}.png'.format(action, sequence))

# #             # Make detections
# #             image, results = mediapipe_detection(frame, model)

# #             # Draw landmarks
# #             image = draw_styled_landmarks(image, results)

# #             # Display messages
# #             if frame_num == 0:
# #                 cv2.putText(image, 'STARTING COLLECTION', (120, 200),
# #                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
# #                 cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
# #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
# #                 cv2.imshow('OpenCV Feed', image)
# #                 cv2.waitKey(200)
# #             else:
# #                 cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
# #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
# #                 cv2.imshow('OpenCV Feed', image)

# #             # Extract keypoints
# #             keypoints = extract_keypoints(results)

# #             # Save keypoints
# #             npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
# #             np.save(npy_path, keypoints)

# #             # Break condition
# #             if cv2.waitKey(10) & 0xFF == ord('q'):
# #                 break

# # cv2.destroyAllWindows()



# numbers = 3934939
# print(numbers.bit_count)


# def rectangle(lenght, width):
#     print("Hello rectangle : ", lenght*width)
# rectangle(2,3)



# def hell(name):
# def hell(*name):
#     # print("my name is  ", name)
#     print("my name is  ", name[0])
#     print("my name is  ", name[1])
#     print("my name is  ", name[2])
# # hell("Ramzia")
# hell("Ramzia", "kamal" ,"jamal")



# return statemt 
# def hello():
#     return "kamal"
# print(hello())



# def add(a,b):
    # return "the addition is : ", a+b
# print(add(1,2))


# recursion inthe python mean when the fucntion call itself
# def hello():
#     print("hello")
#     return hello()
# print(hello())
# def add(x,y):

#     print(x+y)
# add(1,2)




# def fac(n):
#     if n==1:
#         return 1
#     else:
#         return n*fac(n-1)
# print(fac(5))

# num  = 1
# fac = 0
# for i in range(5):
#     if num <i:
#         print(i)
#     else:
#         fac*num-1
# print(fac)



# num = 5
# factorial = 1
# for i in range(1,num+1):
#     factorial*=i
# print(factorial)

# def fac(n):
#     if n==1:
#         return 1
#     else:
#         return n*fac(n-1)
# print(fac(4))


# x = 4
# print("The global one is four", x)
# def hello():
#     global x
#     x = 5
#     return x
# print(hello())
# print(x)



# name = "kamlGlobal"
# print(name)

# def inner():
#     global name
#     name = "kamalInner"
#     return name
# print(inner())
# print(name)




# def con(a,b,c):
#     if a>b and a>c:
#         print("a is greater", a)
#     elif b>a and b>c:
#         print("b is greater", b) 
#     else:
#         print("c is greater", c)
# con(12,23,45)



# def listofSquare():
#     l  =  []
#     for i in range(1,31):
#         l.append(i**2)
#     return l
# # print(listofSquare())


# def check_prime(num):
#     if num <= 1:
#         print(f"{num} is not a prime number")
#         return

#     for i in range(2, int(num**0.5) + 1):
#         if num % i == 0:
#             print(f"{num} is not a prime number")
#             return
#     print(f"{num} is a prime number")


# # Example:
# # check_prime(7)
# # check_prime(10)


# def check_prime(num):
#     if num <= 1:
#         print(num, "is not a prime number")
#         return

#     for i in range(2, int(num**0.5) + 1):
#         if num % i == 0:
#             print(num, "is not a prime number")
#             return
#     print(num, "is a prime number")


# # Example
# check_prime(7)
# check_prime(10)


# def add(numbers):
#     total  =0
#     for i in numbers:
#         total = total+i
#         return total
# print(add([1,2,3,4,5]))




# import datetime
# print(datetime.datetime.now())



# import random
# l = ["kaal", "now"]
# x = random.choice(l)
# print(x)




# import random
# x = random.randint(1,5)
# print(x)



# import random
# names = ["kamal", "jamal", "khan"]
# x = random.choice(names)
# print(x)






# import math
# x = max(1,2,3,4,5)
# print(x)


# b = 5
# x = math.sqrt(81)
# print(x)


# def add(x,y):
#     return x+y


# name = "kamal"
# employee = {"name": "kamal", "age":23}


def hello():
    return "kamla"

