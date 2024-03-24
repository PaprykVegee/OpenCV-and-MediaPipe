import mediapipe as mp
import cv2
import time

class HandDetector():
    def __init__(self,
                 mode=False,
                 maxHands=2):

        self.mode = mode
        self.maxHands = maxHands

        # Tworzenie instancji klasy Hands
        self.mpHands = mp.solutions.hands
        # Wywo³anie metody
        self.hands = self.mpHands.Hands(self.mode,
                                       self.maxHands)
        # rysowanie kropek na palcach
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Konwertowanie obrazu BGR do RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # self.result zmainnna klasy czyli teraz moge jej uzyc w innej funkcji
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Rysowanie punktów na d³oniach
                    # nadpisuje kltke (img) i na kazdej dloni wykrywa
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo = 0, draw = True, double = False):

        # land mark list 
        lmList = []
        if self.results.multi_hand_landmarks:

           
            if double:

               for handIdx, myHand in enumerate(self.results.multi_hand_landmarks):
                   for id, lm in enumerate(myHand.landmark):
                       # Tutaj mo¿esz u¿ywaæ id i lm do operacji na punktach charakterystycznych rêki

                       # pobiera wyskosc, szerokosc, liczba kanalow
                       h, w, c = img.shape

                       # przeskalowanie przez liczbe pikseli na obrazie
                       cx, cy = int(lm.x * w), int(lm.y * h)

                       lmList.append([handIdx, id, cx, cy])

                       if draw:
                           cv2.circle(img,  # na jakim obiekcie
                                       (cx, cy),  # wsplzedna kropka
                                       7,  # promein
                                       (255, 0, 255),  # kolor
                                       cv2.FILLED)  # wypelnienie kropki
            else:

                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                       # Tutaj mo¿esz u¿ywaæ id i lm do operacji na punktach charakterystycznych rêki

                       # pobiera wyskosc, szerokosc, liczba kanalow
                       h, w, c = img.shape

                       # przeskalowanie przez liczbe pikseli na obrazie
                       cx, cy = int(lm.x * w), int(lm.y * h)

                       lmList.append([id, cx, cy])

                       if draw:
                           cv2.circle(img,  # na jakim obiekcie
                                       (cx, cy),  # wsplzedna kropka
                                       7,  # promein
                                       (255, 0, 255),  # kolor
                                       cv2.FILLED)  # wypelnienie kropki

        return lmList
