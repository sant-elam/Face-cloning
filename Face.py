# -*- coding: utf-8 -*-
"""
"""
import cv2 as cv
import mediapipe as mp
import numpy as np

import face_mesh_traingles


def draw_landmarks(image, outputs, land_mark, color, draw=True):
    height, width =image.shape[:2]
            
    points =[]
    for face in land_mark:
        point = outputs.multi_face_landmarks[0].landmark[face]
        
        point_scale = ((int)(point.x * width), (int)(point.y*height))
        
        if draw:
            cv.circle(image, point_scale, 2, color, 1)
        
        points +=[point_scale]
        
    return points
        
def draw_all_face_points(image, outputs, color, draw=True):
    
    height, width =image.shape[:2]
    
    points =[]

    for land_mark_point in outputs.multi_face_landmarks[0].landmark:
        x,y,_ = land_mark_point.x, land_mark_point.y, land_mark_point.z
    
        point_scale = ((int)(x * width), (int)(y*height))
        if draw:
            cv.circle(image, point_scale, 1, color, 1)
        
        points +=[point_scale]
        
        
    return points
             
    
def get_traingle_mesh(source_points, image):
    
    traingle_points_src =[]
    for i in range(0,len(face_mesh_traingles.FACEMESH_TRAINGLE_POINTS), 3):
        pointA = face_mesh_traingles.FACEMESH_TRAINGLE_POINTS[i][0]
        pointB = face_mesh_traingles.FACEMESH_TRAINGLE_POINTS[i+1][0]
        pointC = face_mesh_traingles.FACEMESH_TRAINGLE_POINTS[i+2][0]
               
        len_source_points = len(source_points)
        
        if len_source_points > 0:
            pointA = source_points[pointA]
            pointB = source_points[pointB]
            pointC = source_points[pointC]
            
            traingle_points_src.append([pointA, pointB, pointC])
        
        
    return traingle_points_src



def extract_img_in_triangle(triangle_points, image):
    
    tri_A = triangle_points[0]
    tri_B = triangle_points[1]
    tri_C = triangle_points[2]
    
    triangle_src = np.array([tri_A, tri_B, tri_C], np.int32)
    
    rect = cv.boundingRect(triangle_src)
    
    (x, y, w, h) = rect
    
    
    cropped_image = image[y: y+h, x: x+w]
    
    mask_region = np.zeros((h, w), np.uint8)
    
    points = np.array([[tri_A[0] - x, tri_A[1] - y],
                      [tri_B[0] - x, tri_B[1] - y],
                      [tri_C[0] - x, tri_C[1] - y]], np.int32)
    
    cv.fillConvexPoly(mask_region, points, 255)
    
    try:      
        cropped_trangle = cv.bitwise_and(cropped_image, cropped_image, mask=mask_region)
    except:
        cropped_trangle = cropped_image
        print('Error occur')
    #cropped_trangle = cropped_image
      
    return points, cropped_trangle, rect, mask_region



STATIC_IMAGE = False
MAX_NO_FACES = 2

DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5

COLOR_GREEN = (0,255,0)


#************ LOADING FACE MESH *******************
face_mesh = mp.solutions.face_mesh

face_model = face_mesh.FaceMesh(static_image_mode=STATIC_IMAGE,
                                max_num_faces= MAX_NO_FACES,
                                min_detection_confidence=DETECTION_CONFIDENCE,
                                min_tracking_confidence=TRACKING_CONFIDENCE)


#************ #################### *******************

#***** SOURCE FACE DETECTION ******

source_image = 'source.jpg'

image_src = cv.imread(source_image)
image_src_rgb = cv.cvtColor(image_src, cv.COLOR_BGR2RGB)


outputs = face_model.process(image_src_rgb)

source_points =[]
if outputs.multi_face_landmarks:   
    source_points = draw_all_face_points(image_src, outputs, COLOR_GREEN, False) 
    
traingle_points_src = get_traingle_mesh(source_points, image_src)

#************ #################### *******************

border_offset = 20
top, bottom, left, right = border_offset, border_offset, border_offset, border_offset


#***** VIDEO CAPTURE AND DISPLAY ******
capture = cv.VideoCapture(0)

while True:
    result, image = capture.read()
    
    if result:
        try:
            #-----------Convert image to RGB and Gray---------------- 
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                       
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            #-----------#############################----------------
            
            
            #-----------PROCESSING FACE---------------- 
            outputs = face_model.process(image_rgb)
            #-----------#############################----------------
                    
            
            #-----------Get Face Points ---------------- 
            
            face_mask =[]
            if outputs.multi_face_landmarks:            
                face_mask = draw_all_face_points(image_rgb, outputs, COLOR_GREEN, False) 
            triangle_points_dst = get_traingle_mesh(face_mask,image)  
            
            #-----------#################----------------
            
            
            img_display = image.copy()
            img_new = np.zeros_like(image_rgb, np.uint8) 
            
            
                        
            border_image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0])
            
            if(len(triangle_points_dst) == 0):
                cv.imshow("FACE MESH", border_image)
                if cv.waitKey(30) & 255 == 27:
                    break
                continue
            
            
            
            
            #-----------Mask Off the face ----------------  
            
            points_face = np.array(face_mask, np.int32)
            convex_hull = cv.convexHull(points_face)
            img_display = cv.fillConvexPoly(img_display, convex_hull, 0)
            
            #-----------#################----------------
            
            
            img_head_mask = np.zeros_like(image_gray)
            img_head_mask_ = cv.fillConvexPoly(img_head_mask, convex_hull, 255)
            img_head_mask = cv.bitwise_not(img_head_mask_)
            
            #-----------Iterating through the face traingle points and clone face regions----------------           

            for triangle_dst, triangle_src in zip(triangle_points_dst,traingle_points_src) :
                
                points_src, cropped_src, _, _= extract_img_in_triangle(triangle_src, image_src)
                points_dst, cropped_dst, rect, mask_region= extract_img_in_triangle(triangle_dst, image_rgb)
                
                points_src = np.float32(points_src)        
                points_dst = np.float32(points_dst)
                
                (x, y, w, h) = rect
                
                affine_trans = cv.getAffineTransform(points_src, points_dst)
                warping = cv.warpAffine(cropped_src, affine_trans, (w, h))           
                warping = cv.bitwise_and(warping, warping, mask_region)
                
                
                img_new_reg = img_new[y:y+h, x:x+w]      
                
                img_new_reg_gray = cv.cvtColor(img_new_reg,
                                               cv.COLOR_BGR2GRAY)
                _, mask_triangle = cv.threshold(img_new_reg_gray, 30, 255, cv.THRESH_BINARY_INV)              
                warp_traingle = cv.bitwise_and(warping, warping, mask=mask_triangle) 
                
                img_new_reg = cv.add(img_new_reg, warp_traingle)   
                
                img_new[y:y+h, x:x+w] = img_new_reg
                                   
            result_display = cv.add(img_display, img_new)
            
            #-----------###################################################################----------------
            
            
            #-----------For Seamless Clone----------------
            (x,y,w,h) = cv.boundingRect(convex_hull)           
            center_face = (int((x+x+w)/2)+border_offset,int((y+y+h)/2)+border_offset)
            
            border_image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0])
            border_display = cv.copyMakeBorder(result_display, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0])
            border_mask = cv.copyMakeBorder(img_head_mask_, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)
  
            try :
                seamless_clone = cv.seamlessClone(border_display,
                                                  border_image,
                                                  border_mask, 
                                                  center_face, 
                                                  cv.NORMAL_CLONE)
            except:
                print("")
           
            #-----------********************----------------
            
            
            #-----------Display Face----------------
            cv.imshow("FACE MESH", seamless_clone)
            if cv.waitKey(30) & 255 == 27:
                break
            
            #-----------********************----------------
        except:
         
            #cv.imshow("FACE MESH", border_image)
            #if cv.waitKey(30) & 255 == 27:
                #break
            continue
        
        
capture.release()
cv.destroyAllWindows()
