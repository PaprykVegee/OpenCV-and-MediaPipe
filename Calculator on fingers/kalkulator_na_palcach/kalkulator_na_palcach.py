import mediapipe as mp
import cv2
import time
import HandTracking as ht


def main():

    screen_width = 1920  # dostosuj do swojej rozdzielczości ekranu
    screen_height = 1080  # dostosuj do swojej rozdzielczości ekranu

    cap = cv2.VideoCapture(0) # przechwytywanie obrazu z kamery 0 (gowna kamera lapka)
    cap.set(3, screen_width) # dostoswanie szerokosci
    cap.set(4, screen_height) # dostoswanie wysokosci 

    pTime = 0
    cTime = 0

    # tworzenie obiektu klasy
    handDetector = ht.HandDetector()

    # pkt koncow palcow 
    tipIds = [8, 12, 16, 20]

    while True:

        success, img = cap.read()

        cv2.waitKey(1)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # wywolanie metodyt klasy (odpowiedzailan za detektcje rąk)
        img = handDetector.findHands(img)

        # wywolanie metodyt klasy (odpowiedzailan za pozycje rak i okresle reki dominujacej)
        handNo = 1
        LmList = handDetector.findPosition(img, handNo, draw=True)

        # zwracanie pozycji jednej badz dwoch rak i przypisanie do zmianych lokalnych
        hand_1 = []
        hand_2 = []
        if handNo != 1 and len(LmList) == 2:
            hand_1 = LmList[0]
            hand_2 = LmList[1]
        elif handNo == 1:
            if len(LmList) != 0:
                hand_1 = LmList
                
        # detektcja ile palcow jest pokazywanych
        
        if handNo == 1 and len(hand_1) != 0:
            fingers = []

            # kciuk
            if hand_1[4][2] < hand_1[3][2]:
                hand_state = "Close hand"
                fingers.append(0)
            else:
                hand_state = "Open hand"
                fingers.append(1)

            # reszta poalcow 
            for idx in range(0, 4):
                #print(f"reke 1: {hand_1[idx][3]}")

                hand_state = ""

                if hand_1[tipIds[idx]][3] > hand_1[tipIds[idx] - 2][3]:
                    hand_state = "Close hand"
                    fingers.append(0)
                else:
                    hand_state = "Open hand"
                    fingers.append(1)

            sums = sum(fingers)
            print(sums)
            cv2.putText(img, hand_state, (70, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0), 3)
            cv2.putText(img, str(sums), (70, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0), 3)

        # dodanie textu z fps
        cv2.putText(img,  # gdzie implementujemy
                    f"FPS: {int(fps)}",  # co implementujemy
                    (900, 70),  # gdzie
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,  # czcionka
                    3,  # rozmiar czcionki
                    (255, 0, 0),  # kolor
                    3) # wilkosc

        cv2.imshow("Image", img)

if __name__ == "__main__":
    main()
