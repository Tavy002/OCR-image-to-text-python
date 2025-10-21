import numpy as np
import cv2
import pytesseract

# Se instaleaza tesseract de pe github si apoi se specifica calea acestuia
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# incarca imaginea
image_path = 'farapoza1.jpg'  # inlocuieste cu calea imaginii tale
image = cv2.imread(image_path)

# Convertire la grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicare filtru Gaussian Blur pentru reducerea zgomotului
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Binarizare folosind metoda Otsu
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Detectarea contururilor
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Desenarea contururilor pe imaginea originala
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 10 and h > 10:  # Eliminare zgomot
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Aplicare OCR pentru a extrage textul
detected_text = pytesseract.image_to_string(thresh, lang='eng')
print("Text detectat:", detected_text)

# Afisarea imaginilor preprocesate
cv2.imshow('Original Image with Contours', image)
cv2.imshow('Thresholded Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
