import argparse
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import torch
import sys
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

import math
import mediapipe as mp

class main:
    def __init__(self, arg):
        #Hands detection using MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.which_hand = {0:'Right',1:'Left'}
        self.max_num_hands = arg.max_hands
        self.min_detection_confidence = arg.min_confidence

        self.save_vid = True
        self.video_path = arg.source
        self.font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')

        # initialize StrongSORT
        self.cfg = get_config()
        self.cfg.merge_from_file('strong_sort/configs/strong_sort.yaml')
        self.strong_sort_weights = 'strong_sort/deep/checkpoint/osnet_x0_25_market1501.pth'
        self.device = arg.device

        self.strongsort = StrongSORT(
            self.strong_sort_weights,
            self.device,
            max_dist=self.cfg.STRONGSORT.MAX_DIST,
            max_iou_distance=self.cfg.STRONGSORT.MAX_IOU_DISTANCE,
            max_age=self.cfg.STRONGSORT.MAX_AGE,
            n_init=self.cfg.STRONGSORT.N_INIT,
            nn_budget=self.cfg.STRONGSORT.NN_BUDGET,
            mc_lambda=self.cfg.STRONGSORT.MC_LAMBDA,
            ema_alpha=self.cfg.STRONGSORT.EMA_ALPHA,
        )
        
    def video(self):
        cap = cv2.VideoCapture(self.video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter('output.mp4',
                                 fmt, fps, (width, height))
        tframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        num = 0
        while True:
            print('frame count ' + str(num) + '/' + str(tframe))
            num += 1

            ret, frame = cap.read()
            if not ret:
                break

            outputs,confs,frame = main.any_model(self,frame)
            
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):
                        frame = main.annotation(self, frame, output, conf)
            #save
            if self.save_vid:
                writer.write(frame)
            
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    def any_model(self,frame):
        image = frame
        img_height, img_width, _ = image.shape
        outputs = []
        confs = []

        # Run MediaPipe Hands.
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence) as hands:

            # Convert the BGR image to RGB, flip the image around y-axis for correct 
            # handedness output and process it with MediaPipe Hands.
            results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))

            if results.multi_hand_landmarks:
                score_list = [results.multi_handedness[i].classification[0].score 
                    for i in range(len(results.multi_handedness))]
                label_list = [results.multi_handedness[i].classification[0].label 
                    for i in range(len(results.multi_handedness))]
                # Print handedness (left v.s. right hand).
                #print(results.multi_handedness)

                annotated_image = cv2.flip(image.copy(), 1)

                hands_list = []
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_list = []

                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                    lm = hand_landmarks.landmark[0]
                    lm_xlist = [lm.x for lm in hand_landmarks.landmark]
                    lm_ylist = [lm.y for lm in hand_landmarks.landmark]
                    hand_list.append((1-min(lm_xlist)) * img_width)
                    hand_list.append(min(lm_ylist) * img_height)
                    hand_list.append((1-max(lm_xlist)) * img_width)
                    hand_list.append(max(lm_ylist) * img_height)

                    hands_list.append(hand_list)
                
                #Change annotation coordinates for StrongSORT    
                x = torch.tensor(hands_list)
                xywhs = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
                xywhs[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
                xywhs[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
                xywhs[:, 2] = x[:, 0] - x[:, 2]  # width
                xywhs[:, 3] = x[:, 3] - x[:, 1]  # height

                confs = torch.tensor(score_list)
                clss = [0 if 'Right'==label  else  1 for label in label_list]
                clss = torch.tensor(clss)
                image = cv2.flip(annotated_image, 1)

                #Run StorngSORT
                outputs = self.strongsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), image)         
        return outputs,confs, image

    def annotation(self, frame, output, conf):
        bboxes = output[0:4]
        id = int(output[4])
        clss = int(output[5])
        label = self.which_hand[clss] #Make the object name change to match the clss number

        frame = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
        draw = ImageDraw.Draw(frame)
        rectcolor = (0, 188, 68)
        linewidth = 8
        draw.rectangle([(output[0], output[1]), (output[2], output[3])],
                       outline=rectcolor, width=linewidth)

        textcolor = (255, 255, 255)
        textsize = 40

        #Specify font style by path
        font = ImageFont.truetype(self.font_path, textsize)

        text = f'{id} {label} {conf:.2f}'

        txpos = (output[0], output[1]-textsize-linewidth//2) #Coordinates to start drawing text
        txw, txh = draw.textsize(text, font=font)

        draw.rectangle([txpos, (output[0]+txw, output[1])], outline=rectcolor,
                       fill=rectcolor, width=linewidth)

        draw.text(txpos, text, font=font, fill=textcolor)
        frame = np.asarray(frame)

        return frame
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='your video path')
    parser.add_argument('--device', default='cpu', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--max-hands', default=6, type=int, help='Number of hands to detect')
    parser.add_argument('--min-confidence', default=0.7, type=float, help='min detection confidence')    
    return parser.parse_args(argv)

if __name__ == '__main__':
    arg = parse_arguments(sys.argv[1:])
    run = main(arg)
    run.video()
