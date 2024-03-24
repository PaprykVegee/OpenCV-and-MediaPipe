import mediapipe as mp
import cv2

class Face_Mesh():
    def __init__(self,
                 statickMode = False,
                 maxFaces = 1):

        self.statickMode = statickMode
        self.maxFaces = maxFaces

        # tworzenie instancji klasy face_mash
        self.mpFaceMesh = mp.solutions.face_mesh
        # wywolanie metody 
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.statickMode,
                                                    self.maxFaces)

        # rysowanie kropek mesh na twarzy
        self.mpDraw = mp.solutions.drawing_utils
        # specyfikacja rysowanych elemantow
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = 1, circle_radius = 2)

    def findMeshFace(self, img, draw = True):

        # Konwertowanie obrazu BGR do RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # self.result zmainnna klasy czyli teraz moge jej uzyc w innej funkcji
        self.results = self.faceMesh.process(imgRGB)

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    # Rysowanie punktów na d³oniach
                    # nadpisuje kltke (img) i na kazdej dloni wykrywa
                    self.mpDraw.draw_landmarks(img, faceLms,
                                              self.mpFaceMesh.FACEMESH_TESSELATION,
                                              self.drawSpec, 
                                              self.drawSpec)
        return img

    def findPosition(self, img):
        
        if self.results.multi_face_landmarks:
            # land mark list
            faces = []
            for faceLms in self.results.multi_face_landmarks:
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape

                    x,y = int(lm.x*iw), int(lm.y*ih)
                    face.append([x, y])
            faces.append(face)
        return faces


class Face_detection():
    def __init__(self,
                 statickMode = False,
                 maxFace = 3):
        
         self.statickMode = statickMode
         self.maxFace = maxFace

         # tworzenie instancjji klasy face
         self.mpFace = mp.solutions.face_detection

         # wywolanie metody
         self.face = self.mpFace.FaceDetection(self.statickMode,
                                               self.maxFace)

    def findFace(self, img, draw=True):
            # Konwertowanie obrazu BGR do RGB
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Poprawka: SprawdŸ czy self.result istnieje przed u¿yciem
            self.result = self.face.process(imgRGB)

            if self.result.detections:
                if draw:
                    for detection in self.result.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = img.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(
                            bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                        # Rysowanie prostok¹ta wokó³ wykrytej twarzy
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return img


    def findPosition(self, img):

        if self.results.multi_face_landmarks:
            # land mark list
            faces = []
            for faceLms in self.results.multi_face_landmarks:
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape

                    x,y = int(lm.x*iw), int(lm.y*ih)
                    face.append([x, y])
            faces.append(face)
        return faces
