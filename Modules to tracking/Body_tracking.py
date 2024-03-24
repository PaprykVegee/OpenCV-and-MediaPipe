import mediapipe as mp
import cv2

class BodyDetector():
    def __init__(self,
                 node = False,
                 upBody = False,
                 smooth = True):

        self.node = node
        self.upBody = upBody
        self.smooth = smooth

        # tworzenie instancji klasy pose
        self.mpPose = mp.solutions.pose
        # wywolanie metody 
        self.pose = self.mpPose.Pose(self.node,
                                     self.upBody,
                                     self.smooth)

        # rysowanie kropek mesh na twarzy
        self.mpDraw = mp.solutions.drawing_utils
        # specyfikacja rysowanych elemantow
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = 1, circle_radius = 2)

    @staticmethod
    def calculateBoundingBox(landmarks, ih, iw):

        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

        for lm in landmarks.landmark:
            x, y = int(lm.x * iw), int(lm.y * ih)

            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        return min_x, min_y, max_x, max_y

    def findPose(self, img, draw_point=True, draw_rect=False):

        # Konwertowanie obrazu BGR do RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # self.results to zmienna klasy, któr¹ teraz mo¿na u¿yæ w innej funkcji
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw_point:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

            if draw_rect:
                ih, iw, _ = img.shape
                bbox = self.calculateBoundingBox(self.results.pose_landmarks, ih, iw)

                # Rysowanie prostok¹ta wokó³ wykrytej pozy
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        return img
       