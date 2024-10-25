import face_recognition
import os
import cv2
import urllib.request
import random

#creating path for the folders

directory_1 = "known_faces"
directory_2 = "unknown_faces"
path_1 = os.path.join(directory_1)
os.mkdir(path_1)
path_2 = os.path.join(directory_2)
os.mkdir(path_2)

#getting url from the user and download the train image

def download_image(url, file_path, file_name):
    full_path = file_path + str(file_name) + '.jpg'
    urllib.request.urlretrieve(url, full_path)
    '''url = input('Please enter image URL (string):')'''

url = "https://media-exp1.licdn.com/dms/image/D5603AQEmWOoTKHucyw/profile-displayphoto-shrink_400_400/0/1669100753583?e=1674691200&v=beta&t=-Non-Z_5J4i2QSWeddFvjUlAPke-YkeXA5zMYAy4Igg "

nameID=str(input("Enter Your Name: ")).lower()
path='known_faces/'+nameID
os.path.exists(path)
os.makedirs(path)
file_name=random.randrange(20, 100)

#calling function
download_image(url, 'known_faces/'+nameID+'/', str(file_name))

#getting screenshots from the users
video=cv2.VideoCapture(0)

count=0
#finding whether the folder already exists or not
path='unknown_faces/'
isExist = os.path.exists(path)
if isExist:
    pass
else:
    os.makedirs(path)

#Creating a whille loop for how pictures to be captured by the webcam
while True:
    ret,frame=video.read()
    count=count+1
    name='./unknown_faces/'+ str(count) + '.jpg'
    print("Creating Images........." +name)
    cv2.imwrite(name, frame)
    k=cv2.imshow("WindowFrame", frame)
    cv2.waitKey(1000)
    if count>2:
        break
video.release()
cv2.destroyAllWindows() #moving to the next pictures

#using face recognition
#calling an variable
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.526
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog" #cnn if you have gpu installed
print("loading known faces")

known_faces = []
known_names = []

#the image from the known_faces are loaded and encoded
for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)
print("processing unknown faces")

#unknown_faces images are loaded, located and encoded

for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(filename)
    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    locations = face_recognition.face_locations(image , model =MODEL)
    encodings = face_recognition.face_encodings(image , locations)
    image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)

#comparing the known_faces and unknown_faces to find the match

for face_encoding , face_location in zip(encodings , locations):
    results = face_recognition.compare_faces(known_faces, face_encoding,TOLERANCE)
    match = None
    if True in results :
        match = known_names[results.index(True)]
        print(f"Match found: {match}")
        top_left = (face_location[3] , face_location[0])
        bottom_right = (face_location[1] , face_location[2])
        color = [0,255,0]
        cv2.rectangle(image , top_left , bottom_right , color ,FRAME_THICKNESS)
        top_left = (face_location[3] , face_location[2])
        bottom_right = (face_location[1] , face_location[2]+22)
        cv2.rectangle(image , top_left , bottom_right , color ,cv2.FILLED)
        cv2.putText(image , match , (face_location[3]+10 ,face_location[2]+15) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 0) ,FONT_THICKNESS)
        cv2.imwrite("newimage.jpg", image) # save image
        
captured=cv2.imshow(filename , image)
cv2.waitKey(5000)
cv2.destroyWindow(filename)