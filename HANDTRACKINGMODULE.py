#!/usr/bin/env python
# coding: utf-8

# # $$ Hand Tracking Project $$
# Name: Anubhav Joshi
# </br>
# Indian Institute of Technology Kanpur

# In[1]:


#%% Hand Tracking in real time
'''

 mediapipe guithub by Google
 Hand tracking ----> Palm Detection and Hand Landmark
              Complete image of hand,   21 different landmark manually notated 30K notations of hand

'''

''' Static_image_mode=Defalt False----> images as a video stream.
                                                      first image detect and then 
                                                      localize localize the landmarks
                                                      in same image.
                                                      is set to True, hand detection runs
                                                      on every input image, ideal for processing
                                                      a batch of static, possibly unrelated, images.
    Max_num_Hands=Maximum number of hands to detect.
    Model_complexity= Landmarks accuracy as well as inference lantency generally go up with the model
                      complexity. default=1
    Min_Detection_Confidence=From the hand detection model for the detection to be considered successful.
    Min_Tracking_Confidence= Hand landmarks to be considered tracked successfull or otherwise hand 
                             detection will be invoked autimatically on the next input image.
'''
# Intall opencv-python , mediapipe

# Importing Libraries
import cv2
import mediapipe as mp
import time

# Class hand Detector
class handDetector():
    
    def __init__(self,mode=False,maxHands=2,modelComplexity=1,detectionCon=0.5,trackCon=0.5):

        self.mode=mode
        self.maxHands=maxHands
        
        self.modelComplexity=modelComplexity
        
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        
        # mp.solutions.hands---> import mediapipe librarie and use solutions.hands
        
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity,self.detectionCon, self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils

    
    def findHands(self,img,draw=True):
        
        # Convert image BGR to RGB
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        # Process the hand Image imgRGB
        self.results=self.hands.process(imgRGB)    
        #print(results.multi_hand_landmarks)
    
        if self.results.multi_hand_landmarks:
            
            #Connecting Hand Landmarks
            for handLms in self.results.multi_hand_landmarks:
                
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        
        #Return Hand image with located hand landmarks with connections
        return img
    
    def findPosition(self,img,handNo=0,draw=True):
        
        lmList=[]
        if self.results.multi_hand_landmarks:
            
            myHand=self.results.multi_hand_landmarks[handNo]
            
            for ide,lm in enumerate(myHand.landmark):

                h, w, c=img.shape

                cx,cy=int(lm.x*w),int(lm.y*h)

                #print(ide,cx,cy)
                lmList.append([ide,cx,cy])
                
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)

        return lmList
    
def main():
    # Time for FPS calculation
    pTime=0
    cTime=0
    
    cap=cv2.VideoCapture(0) # Webcam For video 
    detector=handDetector() # Object of Class
    
    while True:
        
        # Read image
        success,img=cap.read()
        
        img=detector.findHands(img)
        
        lmList=detector.findPosition(img)
        
        # Print Location of 4th landmark.
        if len(lmList)!=0:
            print(lmList[4])
        
        # Calculation of Fps
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        
        # Put fps Info. to the Web video with hand detection.
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,
                    3,(255,0,0),3)
    
        # Show Image
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__=='__main__':
    main()


# ---
# ---
