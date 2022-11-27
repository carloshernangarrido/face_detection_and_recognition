import sys
import cv2
import numpy as np

import input_parameters as ip


def main():
    win_name_main = "Face Detection. Q to exit, Enter to autoplay. Press T to train the face recognizer."
    win_name_face = f"Face detected. Press a number to save this face for a specific ID [0-{ip.number_of_people - 1}]."
    win_name_face_gray = win_name_face + " gray"
    cv2.namedWindow(win_name_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_name_face, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(win_name_face_gray, cv2.WINDOW_NORMAL)

    # Face detector
    face_detector = cv2.CascadeClassifier(ip.face_cascade_classifier_filename)

    # Face recognizer
    face_recognizer = cv2.face_LBPHFaceRecognizer()
    face_recognizer = face_recognizer.create()
    people = {'images': [],
              'labels': [],
              'labels_ndarray': None,
              'trained_flag': False}

    # Video Source
    video = None
    if ip.video_source == 'webcam':
        video = cv2.VideoCapture(0)  # for using WebCAM

    # Read the input image
    ok, img = video.read()  # c = frame[y, x, :]
    if not ok:
        print('Cannot read video file')
        sys.exit()

    img_rect, img_face_text, img_face, img_face_gray = \
        img.copy(), img.copy(), img.copy(), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    while True:
        # Read the input image
        ok, img = video.read()  # c = frame[y, x, :]
        if not ok:
            print('Cannot read video file')
            break

        # Detect faces
        faces_ = face_detector.detectMultiScale3(image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                                 scaleFactor=ip.scale_factor,
                                                 minNeighbors=ip.min_neighbours,
                                                 flags=cv2.CASCADE_SCALE_IMAGE,
                                                 outputRejectLevels=True)
        faces = faces_[0]
        weights = faces_[2]
        faces, weights = [_[1] for _ in sorted(zip(weights, faces))], [_[0] for _ in sorted(zip(weights, faces))]
        faces = [wf[1] for wf in sorted(zip(weights, faces)) if wf[0] > ip.detection_threshold]
        weights = [wf[0] for wf in sorted(zip(weights, faces)) if wf[0] > ip.detection_threshold]
        if len(faces) == 0:
            img_rect = cv2.putText(img, f'No faces detected...',
                                   (10, 100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3,
                                   color=(0, 0, 255))
            img_face_text = cv2.putText(img_face, f'Last face detected...',
                                        (5, 100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.75,
                                        color=(0, 0, 255))
        else:
            for (x, y, w, h), weight in zip(faces, weights):
                # Results of face detection
                img_rect = cv2.rectangle(img,
                                         pt1=(x, y), pt2=(x + w, y + h),
                                         color=(0, 255, 0))
                img_rect = cv2.putText(img_rect, f'w = {round(weight)}',
                                       (x, y), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3,
                                       color=(0, 0, 255))
                img_face = img[y:y + h, x:x + w]
                img_face_gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
                img_face_text = img_face

                # Results of face recognition
                if people['trained_flag']:
                    id_dist = face_recognizer.predict(img_face_gray)
                    id_, dist = id_dist[0], round(id_dist[1])

                    dist = dist if dist <= 100 else 100

                    img_rect = cv2.putText(img_rect, f'ID = {id_}',
                                           (x, y+h+80), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4,
                                           color=(0, 255, 255))
                    # img_rect = cv2.putText(img_rect, f'Dist = {dist}',
                    #                        (x, y+h-25), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4,
                    #                        color=(0, 255, 255))
                    img_rect = cv2.rectangle(img_rect,
                                             pt1=(x + w - int(w*dist/100), y + h), pt2=(x + w, y + h + 20),
                                             color=(0, 0, 255), thickness=cv2.FILLED)
                    img_rect = cv2.rectangle(img_rect,
                                             pt1=(x, y + h), pt2=(x + w - int(w*dist/100), y + h + 20),
                                             color=(0, 255, 0), thickness=cv2.FILLED)

        # cv2.imshow(win_name_face_gray, img_face_gray)
        cv2.imshow(win_name_face, img_face_text)
        cv2.imshow(win_name_main, img_rect)

        if ip.flags['auto_play']:
            key = cv2.waitKey(1)
        else:
            key = cv2.waitKey()

        if key & 0xFF == ord('q') or key & 0xFF == ord('Q'):
            break
        elif key == ord('\r'):
            ip.flags['auto_play'] = not ip.flags['auto_play']
        elif (key & 0xFF) in list(map(ord, (list(map(str, range(ip.number_of_people)))))):
            person_id = key - 48
            print("saved person: ", person_id)
            people['images'].append(img_face_gray)
            people['labels'].append(person_id)
        elif key & 0xFF == ord('t') or key & 0xFF == ord('T'):
            n_images = len(people["labels"])
            if n_images > 1:
                print(f'Training with {n_images} images...')
                people['labels_ndarray'] = np.array(people['labels'])
                face_recognizer.train(people['images'], people['labels_ndarray'])
                print("Training complete!")
                people['trained_flag'] = True
            else:
                print("You'll need more than one sample to learn a model")
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
