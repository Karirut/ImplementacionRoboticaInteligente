import cv2

def encontrar_indice_mayor_igual(lista, valor_limite):
    for indice, valor in enumerate(lista):
        if valor >= valor_limite:
            return indice
    return -1

# Cargar y preprocesar las imágenes de referencia
def cargar_y_procesar_imagen(ruta):
    img = cv2.imread(ruta)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img

imagenes = [
    cargar_y_procesar_imagen('C:\\Users\\karir\\OneDrive\\Documents\\Robotica_Codes\\actividad3\\giveaway.jpg'),
    cargar_y_procesar_imagen('C:\\Users\\karir\\OneDrive\\Documents\\Robotica_Codes\\actividad3\\stop.jpg'),
    cargar_y_procesar_imagen('C:\\Users\\karir\\OneDrive\\Documents\\Robotica_Codes\\actividad3\\straight.jpg'),
    cargar_y_procesar_imagen('C:\\Users\\karir\\OneDrive\\Documents\\Robotica_Codes\\actividad3\\turnaround.jpg'),
    cargar_y_procesar_imagen('C:\\Users\\karir\\OneDrive\\Documents\\Robotica_Codes\\actividad3\\turnleft.jpg'),
    cargar_y_procesar_imagen('C:\\Users\\karir\\OneDrive\\Documents\\Robotica_Codes\\actividad3\\turnright.jpg'),
    cargar_y_procesar_imagen('C:\\Users\\karir\\OneDrive\\Documents\\Robotica_Codes\\actividad3\\workinprogress.jpg')
]

nombres = ['giveaway', 'stop', 'straight', 'turn around', 'left', 'right', 'work in progress']
conteo = [0, 0, 0, 0, 0, 0, 0]
iteracion = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Crear objeto SIFT
    sift = cv2.SIFT_create()
    keypoints_ref, descriptors_ref = sift.detectAndCompute(th, None)

    # Crear feature matcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    puntos = []

    # Comparar cada imagen de referencia con la imagen del frame
    for img in imagenes:
        keypoints_img, descriptors_img = sift.detectAndCompute(img, None)
        matches = bf.match(descriptors_img, descriptors_ref)
        puntos.append(len(matches))

    maximo = max(puntos)
    index = puntos.index(maximo)

    conteo[index] += 1
    iteracion += 1
    val = 20

    indice = encontrar_indice_mayor_igual(conteo, val)

    if iteracion > val + 10:
        print('NO HAY SEÑAL')
        iteracion = 0
        conteo = [0, 0, 0, 0, 0, 0, 0]
    else:
        if indice != -1:
            print(nombres[indice])
            conteo = [0, 0, 0, 0, 0, 0, 0]
            iteracion = 0
        else:
            print(conteo)

    cv2.imshow('ref', th)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
