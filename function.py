


# #import dependency
# import cv2
# import numpy as np
# import os
# import mediapipe as mp

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands

# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
#     image.flags.writeable = False                  # Image is no longer writeable
#     results = model.process(image)                 # Make prediction
#     image.flags.writeable = True                   # Image is now writeable 
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
#     return image, results

# def draw_styled_landmarks(image, results):
#     if results.multi_hand_landmarks:
#       for hand_landmarks in results.multi_hand_landmarks:
#         mp_drawing.draw_landmarks(
#             image,
#             hand_landmarks,
#             mp_hands.HAND_CONNECTIONS,
#             mp_drawing_styles.get_default_hand_landmarks_style(),
#             mp_drawing_styles.get_default_hand_connections_style())


# def extract_keypoints(results):
#     if results.multi_hand_landmarks:
#       for hand_landmarks in results.multi_hand_landmarks:
#         rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3)
#         return(np.concatenate([rh]))
# # Path for exported data, numpy arrays
# DATA_PATH = os.path.join('MP_Data') 

# actions = np.array(['A','B','C'])

# no_sequences = 30

# sequence_length = 30


# # import cv2
# # import numpy as np
# # import os
# # from ultralytics import YOLO

# # # Load YOLO Pose model (lightweight)
# # model = YOLO("yolov8n-pose.pt")

# # # This replaces mediapipe_detection()
# # def mediapipe_detection(image, model_yolo):
# #     results = model_yolo(image)
# #     return image, results

# # # This replaces draw_styled_landmarks()
# # def draw_styled_landmarks(image, results):
# #     if results and len(results) > 0:
# #         annotated_image = results[0].plot()  # YOLO's built-in drawing
# #         return annotated_image
# #     return image

# # # This replaces extract_keypoints()
# # def extract_keypoints(results):
# #     if results and len(results) > 0 and results[0].keypoints is not None:
# #         # Shape: (num_persons, num_keypoints, 3)
# #         keypoints = results[0].keypoints.data.cpu().numpy()
# #         # Flatten the first detected person
# #         return keypoints[0].flatten()
# #     else:
# #         return np.zeros(17 * 3)  # YOLOv8 pose has 17 keypoints

# # # Path for exported data
# # DATA_PATH = os.path.join('MP_Data') 
# # actions = np.array(['S', 'R', 'F'])
# # no_sequences = 30
# # sequence_length = 30

# # # Webcam loop (example usage)
# # cap = cv2.VideoCapture(0)
# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     image, results = mediapipe_detection(frame, model)
# #     image = draw_styled_landmarks(image, results)

# #     keypoints = extract_keypoints(results)
# #     print("Keypoints shape:", keypoints.shape)

# #     cv2.imshow("Hand Tracking", image)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()

# import data as d
# # a = data.add(2,3)
# # print(a)

# b = d.employee["name"]

# print(b)

# name = "kamal"


# help(str.format())
count = 0
materials = ["wood", "fire","tea", "cup", "pin"]
for mat in materials:
    count+=1
    print(count,mat)



# count = 0
materials = ["wood", "fire","tea", "cup", "pin"]
for n, mat in enumerate (materials):
    # count+=1
    print(n,mat)




# or use the enmerate 
print(list(enumerate(materials)))





x = 24
y = "hello"
z= [1,2,3]

if type(x)==int:
    print(x**2)
if type(y)==str:
    print("kamal hai bro ")
print(isinstance(x, int))
print(isinstance(y, str))
print(isinstance(z, list))
print(isinstance(x, float))




text = "hello world"
print(len(text))

nmbers = [10,20,30,40,50]
print(len(nmbers))



print(list(range(5)))
print(list(range(5,5)))
print(list(range(5,6,90)))


# x = 12.45454
# y = 190,4279454
# z = 0,45456
# print(x,y,z)
# x_ = round(x, 2)
# y_ = round(y, 1)
# z_ = round(z, 1)
# print(x_, y_, z_)



ma = ['kamla', "jaml", "sadam", "ahad", "ramzia"]
print(sorted(ma))


pro = ["kamal", "jamal", "khan"]
numbers = [1,2,3]
fit = zip(pro, numbers)
fitted = list(fit)
print(fitted)



print(all([True,1,"non"]))
print(all([True,0,"non"]))
print(all([1>0, 5>2]))



print(format("kamal", '^10'))
print(format("kamal", '<10'))


numbers = [1,2,3,4]
doub_number = [x *2 for x in numbers]
print(doub_number)


A = ["a","b","c"]
print("".join(A))



name = "kamal"
print(name.upper())