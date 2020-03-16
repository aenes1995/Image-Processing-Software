import os

from PyQt5.QtWidgets import QFileDialog, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QImage

import cv2
import numpy as np
import random
import copy


class File:
    
    @staticmethod
    def Open(appwindow, pic_label):
        
        fname = QFileDialog.getOpenFileName(appwindow, 'Open file')
        pic_directory= fname[0]        
        pixmap, img= File.Read_Img(pic_directory) 
        if pixmap!= None :
            File.refreshScreen(pic_label, pixmap)
            return img


    @staticmethod
    def Undo(current_image_index):
        pass

    @staticmethod
    def Redo(all_pictures):
        pass
        
    @staticmethod
    def Read_Img(pic_directory):        
        if pic_directory:
            cv_img= cv2.imread(pic_directory, cv2.IMREAD_COLOR)
            #bgr olarak okuyor
            b,g,r= cv2.split(cv_img)   
            cv_img= cv2.merge((r,g,b))#kanalları bölüp rgb oacak şekilde
                                    #tekrar birleştirdik 

            pixmap= File.getPixMap(cv_img)
            return pixmap, cv_img
        else:
            return None

    @staticmethod
    def getPixMap(cv_img):
        if len(np.shape(cv_img))==3:  #rgb ise
            height, width, channel = np.shape(cv_img)    
            bytesPerLine = 3 * width
            qimg= QImage(cv_img, width, height, bytesPerLine, QImage.Format_RGB888)           
            pixmap = QPixmap().fromImage(qimg)

       
        
        return pixmap
    
    @staticmethod
    def refreshScreen(pic_label, pixmap):
        pic_label.resize(pixmap.width(), pixmap.height())
        pic_label.setPixmap(pixmap)
        

class Preprocessing:
    @staticmethod
    def grayLevelTransformation(img):
        gray= img.copy() 
        i=0 
        while i<len(gray):
            j=0
            while j<len(gray[0]):
                k=0
                sum=0
                while k< len(gray[0][0]):
                    sum+= gray[i][j][k]
                    k+=1
                k=0
                while k< len(gray[0][0]):
                    gray[i][j][k]= sum/3
                    k+=1
                j+=1
            i+=1
        return gray

    @staticmethod
    def getResize(img, d1, d2):
        img_tmp= img.copy()
        tmp= np.full([d1, d2], 0)
        
        a= len(img_tmp)
        b= len(img_tmp[0])


        rowFact= d1/len(img_tmp)
        columnFact= d2/len(img_tmp[0])

        for i in range(0, d1):
            for j in range(0, d2):
                row_in_img= i/rowFact
                column_in_img= j/columnFact

                r0= int(np.floor(row_in_img))
                c0= int(np.floor(column_in_img))

                dltRow= row_in_img- r0
                dltColumn= column_in_img- c0

                w00= (1- dltRow) * (1- dltColumn)
                w01= (1- dltRow)* dltColumn
                w10= dltRow* (1- dltColumn)
                w11= dltRow* dltColumn
    
                clr00= img_tmp[r0][c0]
                clr01= img_tmp[r0][c0+1] if (c0+1) < len(img_tmp[0]) else 0                  
                clr10= img_tmp[r0+1][c0] if (r0+1) < len(img_tmp) else 0
                clr11= img_tmp[r0+1][c0+1] if (r0+1)< len(img_tmp) and (c0+1)< len(img_tmp[0]) else 0

                pixel_value= ((w00* clr00)+ (w01* clr01)+ (w10* clr10)+ (w11* clr11))
                tmp[i][j]= pixel_value
                
              
        return tmp       
                                

    @staticmethod
    def Resizing(img, width, length):
        r,g,b= cv2.split(img)
        
        r_processed= Preprocessing.getResize(r, width, length)
        g_processed= Preprocessing.getResize(g, width, length)
        b_processed= Preprocessing.getResize(b, width, length)

        rgb= cv2.merge((r_processed,g_processed,b_processed))
        return rgb

    @staticmethod
    def getCrop(img, r, c, width, length):
        r_end= 0
        c_end= 0
        if r+width< len(img):
            r_end= r+ width
        else:
            r_end= len(img)

        if c+length < len(img[0]):
            c_end= c+ length
        else:
            c_end= len(len(img[0]))

        sub_img= list()
        for i in range(r, r_end):
            row= list()
            for j in range(c, c_end):
                row.append(img[i][j])
            sub_img.append(row)
        return np.asarray(sub_img)

    @staticmethod
    def Crop(img, x, y, width, length):
        r,g,b = cv2.split(img)
        fill= 1
        
        r= Preprocessing.getCrop(r, x, y, width, length)
        g= Preprocessing.getCrop(g, x, y, width, length)
        b= Preprocessing.getCrop(b, x, y, width, length)
    
        return cv2.merge((r,g,b))

    @staticmethod
    def getAllGrayLevels(img):
        grayscales= np.zeros(256)
        for i in range(len(img)):
            for j in range(len(img[0])):
                grayscales[int(img[i][j])]+=1
        return grayscales

    @staticmethod
    def getMax(gray_levels):
        max=0
        for i in gray_levels:
            if gray_levels[i]> max:
                max= gray_levels

        return max

    @staticmethod
    def calculateLenght(gray_level_count, window_row, max):
        if max: return (window_row*gray_level_count)/max
        else: return 0 


    @staticmethod
    def drawIntoHistogram(histogram_window, gray_levels):
        row= len(histogram_window)
        column= len(histogram_window[0])

        max= np.max(gray_levels)
    
        k=5

        for i in range(len(gray_levels)):
            if gray_levels[i]:
                lenght= Preprocessing.calculateLenght(gray_levels[i], row-20, max)
                cv2.line(histogram_window, (k,row), (k, row-int(lenght)),(255,0,0), 1 )
            origin= (k, row)
            k+= 1
        pass
        
         
    @staticmethod        
    def Histogram(img):
        row= 300
        column= 400
        histogram_window= np.full([row, column, 3], 255)

        r,g,b= cv2.split(Preprocessing.grayLevelTransformation(img))
        grayLevels= Preprocessing.getAllGrayLevels(r)

        Preprocessing.drawIntoHistogram(histogram_window, grayLevels)

        cv2.imshow("histogram", histogram_window/255.0)

        
        

def generateMask(kernel_size, fill):
    mask=list()
    
    i=0
    while i<kernel_size:
        tmp= list()
        j=0
        while j<kernel_size:
            tmp.append(fill)
            j+=1
        mask.append(tmp)
        i+=1
    return mask

def get_pixels(i,j,kernel_row_count, kernel_column_count ,img_2d):
    
    pixels= list()    

    row_start= int(i-((kernel_row_count-1)/2))
    row_end= int(i+((kernel_row_count-1)/2))
 
    column_end= int(j+((kernel_column_count-1)/2))

    while row_start<= row_end:
        tmp= list()
        column_start= int(j-((kernel_column_count-1)/2))
        while column_start<= column_end: 
            pixel= img_2d[row_start][column_start]
            tmp.append(pixel)
            column_start+=1
        pixels.append(tmp)
        row_start+=1
    return pixels

def execute_process(img_2d, mask, process):
    processed_img= img_2d.copy()
    
    i= int((len(mask)-1)/2)
    j= int((len(mask)-1)/2)

    row= len(img_2d)-j
    column= len(img_2d[0])-i

    kernel_row_count= len(mask)
    kernel_column_count= len(mask[0])

    while i<row:
        
        j= int((len(mask)-1)/2)
        while j<column:
            
            pixels= get_pixels(i,j, kernel_row_count, kernel_column_count, img_2d) 
            result= process(pixels, mask)
            if result<0:
                processed_img[i][j]= 0
            elif result>255.0:
                processed_img[i][j]= 255
            else:
                processed_img[i][j]= result
                
            j+=1
        i+=1
    return processed_img
    
def normalizeMask(mask):
        sum= np.sum(mask)
        if sum==0 or sum==1:
            return mask
        else:
            for i in range(0,len(mask)):
                for j in range(0,len(mask[0])):
                    mask[i][j]/= sum
            return mask

class Filtering:

    @staticmethod
    def applyMask(pixels,mask):
        r= np.multiply(pixels,mask)
        sum= r.sum()
        return sum

    def applyEdge(pixels,mask):
        r= np.multiply(pixels,mask)
        m= np.mean(r)
        return m
        

    @staticmethod
    def meanFilter(img, kernel_size):
        r,g,b = cv2.split(img)
        
        fill= 1
        mask= generateMask(kernel_size,fill)
        mask= normalizeMask(mask)       

        r= execute_process(r, mask, Filtering.applyMask)
        g= execute_process(g, mask, Filtering.applyMask)
        b= execute_process(b, mask, Filtering.applyMask)
    
        return cv2.merge((r,g,b))

    @staticmethod
    def getMedian(pixels, mask):
        r= np.median(pixels)
        return r
            
    @staticmethod
    def medianFilter(img, kernel_size):
        r,g,b = cv2.split(img)
        fill= 1
        mask= generateMask(kernel_size,fill)
        mask= normalizeMask(mask)

        r= execute_process(r, mask, Filtering.getMedian)
        g= execute_process(g, mask, Filtering.getMedian)
        b= execute_process(b, mask, Filtering.getMedian)
    
        return cv2.merge((r,g,b))

    def sharpenFilter(img, kernel_size):
        r,g,b = cv2.split(img)
        fill= 1
        mask= [[0, -1, 0],[-1, 5, -1],[0, -1, 0]]
        mask= normalizeMask(mask)

        r= execute_process(r, mask, Filtering.applyMask)
        g= execute_process(g, mask, Filtering.applyMask)
        b= execute_process(b, mask, Filtering.applyMask)
    
        return cv2.merge((r,g,b))

    @staticmethod
    def generateLaplaceMask(kernel_size,fill, center ):
        size= kernel_size
        center_index= (kernel_size-1)/2
        mask=list()
        i=0
        while i<size:
            j=0
            tmp=list()
            while j<size:
                if i==center_index and j==center_index:
                    tmp.append(center)
                else:
                    tmp.append(fill)
                j+=1
            mask.append(tmp)
            i+=1
        return mask

    @staticmethod
    def laplacianFilter(img, kernel_size):
        r,g,b = cv2.split(img)
        mask= Filtering.generateLaplaceMask(kernel_size,-1,8)
        mask= normalizeMask(mask)

        r= execute_process(r, mask, Filtering.applyMask)
        g= execute_process(g, mask, Filtering.applyMask)
        b= execute_process(b, mask, Filtering.applyMask)
   
        return cv2.merge((r,g,b))
   
    @staticmethod
    def findingEdges(img, kernel_size):

        tmp= img.copy()

        r_vertical,g_vertical,b_vertical = cv2.split(img)        
        mask_vertical= [[-1,0,1],[-2,0,2],[-1,0,1]]
        mask_vertical= normalizeMask(mask_vertical)

        r_horizontonal,g_horizontonal,b_horizontonal = cv2.split(img)
        mask_horizontonal= [[-1,-2,-1],[0,0,0],[1,2,1]]
        mask_horizontonal= normalizeMask(mask_horizontonal)

        r_vertical= execute_process(r_vertical/255.0, mask_vertical, Filtering.applyMask)
        g_vertical= execute_process(g_vertical/255.0, mask_vertical, Filtering.applyMask)
        b_vertical= execute_process(b_vertical/255.0, mask_vertical, Filtering.applyMask)

        #cv2.imshow("as", cv2.merge((r_vertical, g_vertical, b_vertical)))

        r_horizontonal= execute_process(r_horizontonal/255.0, mask_horizontonal, Filtering.applyMask)
        g_horizontonal= execute_process(g_horizontonal/255.0, mask_horizontonal, Filtering.applyMask)
        b_horizontonal= execute_process(b_horizontonal/255.0, mask_horizontonal, Filtering.applyMask)

        r= np.sqrt(np.power(r_vertical,2)+ np.power(r_horizontonal,2))
        g= np.sqrt(np.power(g_vertical,2)+ np.power(g_horizontonal,2))
        b= np.sqrt(np.power(b_vertical,2)+ np.power(b_horizontonal,2))

        tmp= np.asarray(cv2.merge((r,g,b)))
        return tmp

        
def getStateValueOfMask(img,i,j,mask):
    state_value= -1
    
    row_start= int(i-((len(mask)-1)/2))
    row_end= int(i+((len(mask)-1)/2))

    column_start= int(j-((len(mask[0])-1)/2))
    column_end= int(j+((len(mask[0])-1)/2))

    count_ones=0
    count_trues= 0

    centerHit= False

    for r in range(row_start,row_end+1):
        for c in range(column_start, column_end+1):
            pixel= img[r][c]
            mask_pixel= mask[int(r- row_start)][int(c- column_start)]
            if mask_pixel== 1: 
                count_ones+=1
                if pixel and 1: 
                    count_trues+=1  #255 and 1= 1
                    if r==i and c==j: centerHit= True
                    
    if count_trues== count_ones: state_value= 1 #fit
    elif count_trues!=0 and centerHit== True: state_value= 0   #hit
    else : state_value= -1   #miss

    return state_value


class Morphological_Operations:
    
    @staticmethod
    def getDilation(processed_img, i, j, mask, state_value): #içeriğini değiştirmesi
                                                             #gerekiyor
        row_start= int(i-((len(mask)-1)/2))
        row_end= int(i+((len(mask)-1)/2))

        column_start= int(j-((len(mask[0])-1)/2))
        column_end= int(j+((len(mask[0])-1)/2))
                                                
        if state_value== 0:  #hit

            for r in range(row_start, row_end+1):
                for c in range(column_start, column_end+1):
                    a= mask[int(r- row_start)][int(c- column_start)] 
                    if a: processed_img[r][c]= 255
                    
    @staticmethod
    def getErosion(processed_img, i, j, mask, state_value): #içeriğini değiştirmesi
                                                             #gerekiyor
        row_start= int(i-((len(mask)-1)/2))
        row_end= int(i+((len(mask)-1)/2))

        column_start= int(j-((len(mask[0])-1)/2))
        column_end= int(j+((len(mask[0])-1)/2))
                                                
        if state_value== 0:  #hit

            for r in range(row_start, row_end+1):
                for c in range(column_start, column_end+1):
                    a= mask[int(r- row_start)][int(c- column_start)] 
                    if a: processed_img[r][c]= 0

    @staticmethod
    def execute_morphological_process(img_2d, mask, morphological_process):
        processed_img= np.copy(img_2d)

        i= int((len(mask)-1)/2)
        j= int((len(mask)-1)/2)

        row= int(len(img_2d)-j)
        column= int(len(img_2d[0])-i)

        while i<row:
        
            j= int((len(mask)-1)/2)
            while j<column:
            
                state_value= getStateValueOfMask(img_2d, i, j, mask)
                morphological_process(processed_img, i, j, mask, state_value) 
                
                j+=1
            i+=1
        return processed_img    


    @staticmethod
    def Dilation(img, kernel_size): 
        
        r,g,b= cv2.split(img)       
        mask= [[0,1,0],[1,1,1],[0,1,0]]

        r= Morphological_Operations.execute_morphological_process(r,mask, Morphological_Operations.getDilation)
       

        return cv2.merge((r,r,r))
        
    @staticmethod
    def Erosion(img, kernel_size):
       
        r,g,b= cv2.split(img)
        mask= [[0,1,0],[1,1,1],[0,1,0]]

        r= Morphological_Operations.execute_morphological_process(r,mask, Morphological_Operations.getErosion)
       
        return cv2.merge((r,r,r))

    @staticmethod
    def Opening():
        pass

    @staticmethod
    def Closing():
        pass



class Center:
    def __init__(self):
        self.center= [0.0, 0.0, 0.0]
        self.sum_of_points= [0.0, 0.0, 0.0]
        self.point_count= 0

    def addSumOfPoints(self, point):
        for i in range(len(self.sum_of_points)):
            self.sum_of_points[i]+= point[i]
        self.point_count+= 1

    def flushPoints(self):
        self.sum_of_points= [0.0, [0.0], [0.0]]
        self.point_count= 0


class Segmentation:

    @staticmethod
    def getOtsuThresholdValue(img_channel):

        grayLevels= Preprocessing.getAllGrayLevels(img_channel)
        pixel_count= len(img_channel)* len(img_channel[0])
        totalMean= 0.0 

        for i in range(256):totalMean += i * grayLevels[i];

        threshold = 0        
        background_pixel = 0
        background_sum = 0.0
        max_variance = 0.0

        for i in range(256):

            background_pixel += grayLevels[i];
            if (background_pixel == 0): continue
            elif(background_pixel == pixel_count): break

            foreground_sum = pixel_count - background_pixel
            background_sum += i * grayLevels[i];
            background_mean = background_sum / background_pixel;
            foreground_mean = (totalMean - background_sum) / foreground_sum;

            variance = float(background_pixel * foreground_sum) * np.power((background_mean - foreground_mean), 2);
            if(variance > max_variance):
                max_variance = variance;
                threshold = i

        return threshold

    @staticmethod
    def getThreshold(img_channel, threshold_value):
        tmp= img_channel.copy()

        for i in range(0, len(img_channel)):
            for j in range(0, len(img_channel[0])):
                tmp[i][j]= 255 if img_channel[i][j]>= threshold_value else 0                 
        return tmp

    @staticmethod
    def OtsuThresholding(img):
        r,g,b= cv2.split(img)
        
        
        k= Segmentation.getOtsuThresholdValue(r)
        r= Segmentation.getThreshold(r, k)

        k= Segmentation.getOtsuThresholdValue(g)
        g= Segmentation.getThreshold(g, k)
        

        k= Segmentation.getOtsuThresholdValue(b)
        b= Segmentation.getThreshold(b, k)

        return cv2.merge((r,g,b))
    
    @staticmethod
    def Thresholding(img, threshold_value):
        r,g,b= cv2.split(img)
        
        r= Segmentation.getThreshold(r, threshold_value)
        g= Segmentation.getThreshold(g, threshold_value)
        b= Segmentation.getThreshold(b, threshold_value)

        return cv2.merge((r,g,b))

    @staticmethod
    def getRandomCenters(img, k):
        centers= list()
        for i in range(k):
            cntr= Center()
            tmp= [random.randint(0,255) for i in range(3)]            
            cntr.center= (tmp)
            centers.append(cntr)
        return centers

    @staticmethod
    def getDistance(p1, p2):
        sum=0       
        for i in range(len(p1)):
            sum+= np.power((p1[i]-p2[i]), 2)
        return np.sqrt(sum)

    @staticmethod
    def getNearestCenter(centers, pixel):
        current_center_index= 0
        p= centers[0].center
        min_distance= Segmentation.getDistance(p, pixel)
        for i in range(len(centers)):
            center= centers[i].center
            dist= Segmentation.getDistance(center, pixel)
            if dist< min_distance:                
                min_distance= dist
                current_center_index= i

        return current_center_index            

    @staticmethod
    def includePointsToNearestCenter(img, centers, center_indexes):
        r= len(img)
        c= len(img[0])
        for i in range(r):
            for j in range(c):
                pixel= img[i][j]
                cntr_index= Segmentation.getNearestCenter(centers, pixel)                
                centers[cntr_index].addSumOfPoints(pixel)
                center_indexes[i][j]= cntr_index
        return centers

    @staticmethod
    def generateNewCenterValue(center):

        center.addSumOfPoints(center.center)
        value= [0.0, 0.0, 0.0]       
        for i in range(len(center.center)):
            if center.point_count!= 0: value[i]= int(center.sum_of_points[i]/ center.point_count)
            
        center.center= value
        center.flushPoints()           
        return center

    @staticmethod
    def calculateNewCenters(centers):
        tmp = copy.deepcopy(centers)      
        result= list()

        for i in range(len(tmp)):            
            new_center_value= Segmentation.generateNewCenterValue(tmp[i])
            result.append(new_center_value)

        return tmp
                             
    @staticmethod
    def Changed(old_centers, new_centers):
        changed= False
        for i in range(len(old_centers)):
            if old_centers[i].center!= new_centers[i].center:
                changed= True
                return changed
        return changed
    
        
    @staticmethod
    def kMeans(img, k):

        tmp_img= img.copy()

        centers= Segmentation.getRandomCenters(tmp_img,k)
        center_indexes= np.zeros([len(tmp_img),len(tmp_img[0])])

        old_centers= copy.deepcopy(centers)
       

        for i in range(100):
            old_centers= Segmentation.includePointsToNearestCenter(tmp_img, old_centers, center_indexes)
            new_centers= Segmentation.calculateNewCenters(copy.deepcopy(old_centers))

            bool= Segmentation.Changed(old_centers, new_centers)
            if bool == False:
                break
            
            for j in range(k):
                old_centers[j].center= (new_centers[j].center)
            print(i)
        
        for i in range(len(tmp_img)):
            for j in range(len(tmp_img[0])):
                tmp_img[i][j]= centers[int(center_indexes[i][j])].center

        return tmp_img
                
        
        
        
        
