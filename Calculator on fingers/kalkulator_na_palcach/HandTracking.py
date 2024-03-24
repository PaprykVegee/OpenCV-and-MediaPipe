import mediapipe as mp
import cv2
import time

class HandDetector():
    def __init__(self, mode=False, maxHands=2):
        self.mode = mode
        self.maxHands = maxHands

        # Tworzenie instancji klasy Hands
        self.mpHands = mp.solutions.hands
        # Wywo³anie metody
        self.hands = self.mpHands.Hands(self.mode, self.maxHands)
        # rysowanie kropek na palcach
        self.mpDraw = mp.solutions.drawing_utils

        # Lista wyników dla wszystkich d³oni
        self.results = None

    def findHands(self, img, draw=True):
        # Konwertowanie obrazu BGR do RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # self.results zmienna klasy, teraz mogê jej u¿yæ w innej funkcji
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Rysowanie punktów na d³oniach
                    # nadpisuje klatkê (img) i na ka¿dej d³oni wykrywa
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=1, draw=True):
        lmLists = []

        if self.results.multi_hand_landmarks:
            for handIdx, myHand in enumerate(self.results.multi_hand_landmarks):
                lmList = []
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([handIdx, id, cx, cy])

                    if draw and ((handNo == 1 and handIdx == 0) or handNo != 1):
                        cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

                lmLists.append(lmList)

        if handNo == 1:
            lmLists = lmLists[0] if lmLists else []

        return lmLists
