import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import cv2
import numpy as np
import scipy.ndimage
from PIL import Image, ImageDraw
from tqdm import tqdm
import mediapipe as mp
import argparse


def image_align(img, face_landmarks, output_size=256,
        transform_size=512, enable_padding=True, x_scale=1,
        y_scale=1, em_scale=0.1, alpha=False, pad_mode='const'):
  lm = np.array(face_landmarks)
  lm[:,0] *= img.size[0]
  lm[:,1] *= img.size[1]
  lm_eye_right      = lm[0:16]  
  lm_eye_left     = lm[16:32]  
  lm_mouth_outer   = lm[32:]  
  # lm_mouth_inner   = lm[60 : 68]  # left-clockwise
  lm_mouth_outer_x = lm_mouth_outer[:,0].tolist()
  left_index = lm_mouth_outer_x.index(min(lm_mouth_outer_x))
  right_index = lm_mouth_outer_x.index(max(lm_mouth_outer_x))
  # print(left_index,right_index)
  # Calculate auxiliary vectors.
  eye_left     = np.mean(lm_eye_left, axis=0)
  # eye_left[[0,1]] = eye_left[[1,0]]
  eye_right    = np.mean(lm_eye_right, axis=0)
  # eye_right[[0,1]] = eye_right[[1,0]]
  eye_avg      = (eye_left + eye_right) * 0.5
  eye_to_eye   = eye_right - eye_left
  # print(lm_mouth_outer)s
  mouth_avg    = (lm_mouth_outer[left_index,:] + lm_mouth_outer[right_index,:])/2.0
  # mouth_avg[[0,1]] = mouth_avg[[1,0]]
  
  eye_to_mouth = mouth_avg - eye_avg
  # Choose oriented crop rectangle.
  x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
  x /= np.hypot(*x)
  x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
  x *= x_scale
  y = np.flipud(x) * [-y_scale, y_scale]
  c = eye_avg + eye_to_mouth * em_scale
  quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
  qsize = np.hypot(*x) * 2

  # Shrink.
  shrink = int(np.floor(qsize / output_size * 0.5))
  if shrink > 1:
    rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
    img = img.resize(rsize, Image.ANTIALIAS)
    quad /= shrink
    qsize /= shrink

  # Crop.
  border = max(int(np.rint(qsize * 0.1)), 3)
  crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
  crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
  if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
    img = img.crop(crop)
    quad -= crop[0:2]

  # Pad.
  pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
  pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
  if enable_padding and max(pad) > border - 4:
    pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
    if pad_mode == 'const':
      img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'constant', constant_values=0)
    else:
      img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
    h, w, _ = img.shape
    y, x, _ = np.ogrid[:h, :w, :1]
    mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
    blur = qsize * 0.02
    img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
    img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
    img = np.uint8(np.clip(np.rint(img), 0, 255))
    if alpha:
      mask = 1-np.clip(3.0 * mask, 0.0, 1.0)
      mask = np.uint8(np.clip(np.rint(mask*255), 0, 255))
      img = np.concatenate((img, mask), axis=2)
      img = Image.fromarray(img, 'RGBA')
    else:
      img = Image.fromarray(img, 'RGB')
    quad += pad[:2]
  img = img.transform((transform_size, transform_size), Image.Transform.QUAD,
            (quad + 0.5).flatten(), Image.Resampling.BILINEAR)
  out_image = img.resize((output_size, output_size), Image.Resampling.LANCZOS)
  out_image = img
  return out_image

def get_aligned_image(image_rgb, results, verbose=False):
  FACEMESH_LIPS = [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                            (17, 314), (314, 405), (405, 321), (321, 375),
                            (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                            (37, 0), (0, 267),
                            (267, 269), (269, 270), (270, 409), (409, 291),
                            (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                            (14, 317), (317, 402), (402, 318), (318, 324),
                            (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                            (82, 13), (13, 312), (312, 311), (311, 310),
                            (310, 415), (415, 308)]
  FACEMESH_LEFT_EYE = [(263, 249), (249, 390), (390, 373), (373, 374),
                                (374, 380), (380, 381), (381, 382), (382, 362),
                                (263, 466), (466, 388), (388, 387), (387, 386),
                                (386, 385), (385, 384), (384, 398), (398, 362)]
  FACEMESH_RIGHT_EYE = [(33, 7), (7, 163), (163, 144), (144, 145),
                                  (145, 153), (153, 154), (154, 155), (155, 133),
                                  (33, 246), (246, 161), (161, 160), (160, 159),
                                  (159, 158), (158, 157), (157, 173), (173, 133)]
  Left_eye = []
  Right_eye = []
  Lips = []
  for (x,y) in FACEMESH_LEFT_EYE:
    if x not in Left_eye:
      Left_eye.append(x)
    if y not in Left_eye:
      Left_eye.append(y)
  for (x,y) in FACEMESH_RIGHT_EYE:
    if x not in Right_eye:
      Right_eye.append(x)
    if y not in Right_eye:
      Right_eye.append(y)
  for (x,y) in FACEMESH_LIPS:
    if x not in Lips:
      Lips.append(x)
    if y not in Lips:
      Lips.append(y)
  landmark_dict = {}
  #results = face_mesh.process(image_rgb)
  #if results == None:
  #      print("Processing face mesh had some issue...")
  #if not results.face_landmarks[0]: #results.multi_face_landmarks:
  #      raise RuntimeError(f"No face landmarks")
  #img_h, img_w, img_c = image_rgb.shape
  #face_3d = []
  #face_2d = []
  #for face_landmarks in results.face_landmarks[0]: #results.multi_face_landmarks:
  #    landmark_dict = restructure_landmark_mediapipe(face_landmarks) #.landmark)
  #    for idx, lm in enumerate(face_landmarks): #.landmark):
  #      if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
  #          x, y = int(lm.x * img_w), int(lm.y * img_h)
  #          # Get the 2D Coordinates
  #          face_2d.append([x, y])
            # Get the 3D Coordinates
  #          face_3d.append([x, y, lm.z])
  #if results is None or not results.face_landmarks:
  #    raise RuntimeError("No face detected")
  # Only one face because num_faces=1
  face_landmarks = results.face_landmarks[0]
  img_h, img_w, img_c = image_rgb.shape
  face_3d = []
  face_2d = []
  landmark_dict = {}
  for idx, lm in enumerate(face_landmarks):
      x = int(lm.x * img_w)
      y = int(lm.y * img_h)
      z = lm.z
      face_2d.append([x, y])
      face_3d.append([x, y, z])
      landmark_dict[idx] = (x, y)
  # Convert it to the NumPy array
  face_2d = np.array(face_2d, dtype=np.float64)
  # Convert it to the NumPy array
  face_3d = np.array(face_3d, dtype=np.float64)
  # The camera matrix
  focal_length = 1 * img_w
  cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])
  # The distortion parameters
  dist_matrix = np.zeros((4, 1), dtype=np.float64)
  # Solve PnP
  success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
  # Get rotational matrix
  rmat, jac = cv2.Rodrigues(rot_vec)
  # Get angles
  angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
  # Get the y rotation degree
  x = angles[0] * 360
  y = angles[1] * 360
  z = angles[2] * 360
  lm_left_eye_x = []
  lm_left_eye_y = []
  lm_right_eye_x = []
  lm_right_eye_y = []
  lm_lips_x = []
  lm_lips_y = []
  for i in Left_eye:
        lm_left_eye_x.append(face_landmarks[i].x)#.landmark[i].x)
        lm_left_eye_y.append(face_landmarks[i].y)#.landmark[i].y)
  for i in Right_eye:
        lm_right_eye_x.append(face_landmarks[i].x)#.landmark[i].x)
        lm_right_eye_y.append(face_landmarks[i].y)#.landmark[i].y)
  for i in Lips:
        lm_lips_x.append(face_landmarks[i].x)#.landmark[i].x)
        lm_lips_y.append(face_landmarks[i].y)#.landmark[i].y)
  lm_x = lm_left_eye_x + lm_right_eye_x + lm_lips_x
  lm_y = lm_left_eye_y + lm_right_eye_y + lm_lips_y
  landmark = np.array([lm_x,lm_y]).T
  # At the end of get_aligned_image function
  aligned_image = image_align(Image.fromarray(image_rgb), landmark)
  aligned_rgb = np.array(aligned_image)  # Convert back to NumPy array
  return aligned_rgb

if __name__ == "__main__":
    aligned_image_rgb = get_aligned_image(image_rgb, results)
