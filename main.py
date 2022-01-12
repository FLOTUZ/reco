import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def leerVideo(rutaVideo: str):
    # LEER VIDEOS DE CAMARA
    # ID_CAMARA = 0
    # camaraWeb = cv.VideoCapture(ID_CAMARA)

    # LEER VIDEO DE ARCHIVO
    video = cv.VideoCapture(rutaVideo)

    while True:
        isTrue, frame = video.read()
        cv.imshow('ORIGINAL', frame)

        # Filtro blanco y negro
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 85, 255, cv.THRESH_BINARY)

        cv.imshow('Filtro', thresh)

        # Tecla para terminar la lectura de video
        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    video.release()
    cv.destroyAllWindows()
    pass


def dibujarEnVideo():
    # Dibujar rectangulo
    img = np.zeros((500, 500, 3), dtype='uint8')
    cv.rectangle(img, (0, 0), (250, 250), (0, 255, 0))

    # Dibujar texto
    cv.putText(img, "mouse", (255, 255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)

    cv.imshow('Rectangulo', img)
    cv.waitKey(0)
    pass


def filtroAlVideo():
    img = cv.imread('imagenes/yo.jpg')
    bordes = cv.Canny(img, 150, 200)
    cv.imshow('Filtro', bordes)

    cv.waitKey(0)
    pass


def treshhold():
    img = cv.imread('imagenes/yo.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshhold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
    cv.imshow('Filtro', thresh)
    cv.waitKey(0)
    pass


def track():
    # Se leé el video
    video = cv.VideoCapture("videos/mouse.mp4")

    # Escala en X
    plt.xlim(0, 600)

    # Escala en y
    plt.ylim(250, 0)

    while True:
        _, frame = video.read()

        # Se selecciona area que nos interesa del video
        # [y1:y2 , x1:x2]
        areaDeInteres = frame[60: 400]

        # Se le aplica filtro blanco y negro
        gray = cv.cvtColor(areaDeInteres, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)

        # Se buscan contornos en la imagen resultante
        contornos, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            # Calcular area para dibujar rectangulo
            area = cv.contourArea(cnt)

            if area > 100:
                # Dibuja contorno del objeto detectado
                cv.drawContours(areaDeInteres, [cnt], -1, (0, 255, 0), 2)

                # Dibuja el rectangulo dependiendo del contorno
                x, y, w, h = cv.boundingRect(cnt)
                cv.rectangle(areaDeInteres, (x, y), (x + w, y + h), (0, 255, 0), 3)

                plt.scatter(x, y, s=10)

        # Se muestra el video con el traking del objeto
        cv.imshow("final", areaDeInteres)

        if cv.waitKey(20) & 0xFF == ord('d'):
            plt.show()
            break
    video.release()
    cv.destroyAllWindows()

    pass


if __name__ == '__main__':
    # leerVideo('videos/mouse.mp4')
    # dibujarEnVideo()
    # filtroAlVideo()
    # treshhold()
    track()
