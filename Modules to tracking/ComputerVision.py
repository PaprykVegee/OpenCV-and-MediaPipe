import mediapipe as mp
import cv2
import time
import Hand_Tracking as ht
import Face_Mesh_tracking as fmt
import Body_tracking as bt


def main():
    pTime = 0  # previos time
    cTime = 0  # current time

    # Wybór kamery, która bêdzie u¿ywana
    cap = cv2.VideoCapture(0)

    # Dostosuj rozmiary okna
    screen_width = 1920  # dostosuj do swojej rozdzielczoœci ekranu
    screen_height = 1080  # dostosuj do swojej rozdzielczoœci ekranu

    cap.set(3, screen_width)
    cap.set(4, screen_height)

    # Tworzenie obiektu kalsy HandDetector
    hands_detector = ht.HandDetector()

    # Tworzenie obiektu klasy Face_Mesh
    faces_mesh_detector = fmt.Face_Mesh()

    # Tworzenie obiektu klasy Face_detection
    faces_detector = fmt.Face_detection()

    # Tworzenie obiektu klasy BodyDetector
    body_Detector = bt.BodyDetector()

    while True:
        # Odczytuje klatkê z kamery
        success, img = cap.read()

        # Check if the image dimensions are valid
        if not success or img.shape[0] == 0 or img.shape[1] == 0:
            # If the read was not successful or the image dimensions are zero, continue to the next iteration
            continue

        # wywlanie metody klasy i przypisane jej do zmienej (nadpisnie zmiennej)
        img = hands_detector.findHands(img, draw = True)

        # wywolanie mewtody klasu i przpisanie jeje zminnej (nadpisanie)
        img = body_Detector.findPose(img, draw_point = True, draw_rect= False)

        # wywolanie metody klasy i przypisanie jej zmiennej (nadpisanie)
        img = faces_mesh_detector.findMeshFace(img, draw=False)

        # wywolanie metody klasy i przypisanie jej zmiennej (nadpisanie)
        img = faces_detector.findFace(img, draw=True)

        # wywolnie metody klasy ktora zwraca pozycje id reki 
        position = hands_detector.findPosition(img, handNo=0, draw=False, double=False)
        if len(position) > 0:
            pass
            #print(position[4])
        


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img,  # gdzie implementujemy
                    str(int(fps)),  # co implementujemy
                    (10, 70),  # wielkosc
                    cv2.FONT_HERSHEY_COMPLEX,  # czcionka
                    3,  # rozmiar czcionki
                    (255, 0, 255),  # kolor
                    3)

        # Wyœwietla klatkê obrazu w oknie o nazwie "Image"
        # to musi byc pod petla zeby napisalo wyswietlana klatke
        cv2.imshow("Image", img)

        # Wartoœæ w nawiasie 0 oznacza, ¿e trzeba coœ klikn¹æ, ¿eby dzia³a³o, a jak jest 1 to dzia³a w sposób ci¹g³y
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
