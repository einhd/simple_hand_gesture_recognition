
import numpy as np
import cv2
import math


while capture.isOpened():

    ret, frame = capture.read()                                                                                         #Kameradan kareleri alma komutu
    cv2.putText(frame, "Elinizi buraya getiriniz.", (120, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 0)                                                        #Belirtilen ölçü ve pozisyondaki görüntüyü al
    crop_image = frame[100:300, 100:300]

    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)                                                                      #Blur uygula

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)                                                                         #Maskeleme ve ayırma işlerini daha kolay yapabilmek için, RGB renk uzayından
                                                                                                                        #HSV (Renk özü, doygunluk, parlaklık) renk uzayına çeviriyoruz.

    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))                                             # Cilt rengini beyaz, diğer tüm renkleri siyah olacak şekilde maskeleme

    kernel = np.ones((5, 5))                                                                                            #Morfolojik Dönüşüm için

    dilation = cv2.dilate(mask2, kernel, iterations=1)                                                                  #Maskelenmiş siyah beyaz görüntüde, gürültüyü azaltmak için erotion ve dilation işlemi
    erosion = cv2.erode(dilation, kernel, iterations=1)                                                                 #Erosion : Sınırlardan genişletme, dilation daraltma işlemidir.


    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    cv2.imshow("Thresholded", thresh)                                                                                   #Ayrılmış görüntüyü göster

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)                              #Cilt renginde olan nesnenin; kontur, çevresini bul.

    try:

        contour = max(contours, key=lambda x: cv2.contourArea(x))                                                       #Alanı en büyük olan nesnenin Konturunu bul, yani kameraya ikinci bir el sokulduğunda
                                                                                                                        #Thresholdda gözükecektir fakat, kontur çizilen yerde görünmeyecek, sonuç için dikkate alınmayacaktır.

        x, y, w, h = cv2.boundingRect(contour)                                                                          #Çizilen konturun etrafını takip edecek şekilde bir çokgen çiz
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        hull = cv2.convexHull(contour)                                                                                  #Şekli kapsayan mümkün en küçük çokgen


        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (150, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (100, 0, 255), 0)

        hull = cv2.convexHull(contour, returnPoints=False)                                                              #Dışbükey
        defects = cv2.convexityDefects(contour, hull)                                                                   #İçbükey, parmak aralarını sayarak sayıyı tespit etmek için kullanacağız

        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])                                                                                #Başlangıç,yakın ve uzak noktaları tanımladık
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
                                                                                                                        #Başlangıç ve bitiş noktasından uzak noktaların açısını, yani dışbükey noktaları (parmak ucu gibi)
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)                                          #kosinüs teoremini kullanıyoruz.
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            if angle <= 90:                                                                                             #Eğer açı 90'dan küçükse uzak noktaya nokta koy, ve sayacı bir arttır.
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

            cv2.line(crop_image, start, end, [0, 255, 150], 2)

        if count_defects == 0:                                                                                          #Sayaca, yani bulunan dar açılara göre, sonucu yazdır.
            cv2.putText(frame, "BIR - 1", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(150,150,255),2)
        elif count_defects == 1:
            cv2.putText(frame, "IKI - 2", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(100,100,255), 2)
        elif count_defects == 2:
            cv2.putText(frame, "UC - 3", (5, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(50,50,255), 2)
        elif count_defects == 3:
            cv2.putText(frame, "DORT - 4", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(200,200,255), 2)
        elif count_defects == 4:
            cv2.putText(frame, "BES - 5", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(0,250,255), 2)
        else:
            pass
    except:
        pass

    cv2.imshow("Kamera", frame)
    all_image = np.hstack((drawing, crop_image))
    cv2.imshow('Kontur', all_image)

    if cv2.waitKey(1) == ord('k'):                                                                                      #"k" harfine basıldığında uygulamayı kapat
        break

capture.release()                                                                                                       #Kamerayı kapat
cv2.destroyAllWindows()                                                                                                 #Bütün pencereleri kapat