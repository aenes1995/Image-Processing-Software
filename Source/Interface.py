import sys
import os

from PyQt5.QtWidgets import QWidget, QMainWindow, QMenu, QAction, QLabel, QLineEdit, QPushButton                           
from PyQt5.QtGui import QIcon

import numpy as np
import matplotlib.pyplot as plt
import cv2

import scipy

import Tasks


class App_Window(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.title = 'Image Processing Toolbox'
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 400


        self.all_pictures= list(np.array([]))
        self.current_image_index= 0
        self.have_worked= False
        
        self.pic_label= QLabel(self)
        self.pic_label.move(25,50)        

        self.initUI()
        self.initMenuBar()

        self.show()
        

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

       

    def initMenuBar(self):
        menuBar = self.menuBar()
        
        self.createMenu(menuBar,"File",["Open","Save", "Undo"])

        self.createMenu(menuBar,"Preprocessing",
                ["Gray Level Transformation", "Resizing", "Zoom", "Crop", "Histogram"])

        self.createMenu(menuBar,"Filtering",["Mean Filter","Median Filter",
                "Sharpen","Laplace Filter","Finding Edges"])  

        self.createMenu(menuBar,"Morphological Operations",["Dilation","Erosion",
                "Opening","Closing"])

        self.createMenu(menuBar,"Segmantation",["Thresholding", "Otsu Thresholding" ,"K- Means", "Neighbourhood" ])      
        

    def createMenu(self, menuBar, menuName, menuItems):

        new_menu= menuBar.addMenu("&"+menuName)
        for item in menuItems:            
            action= new_menu.addAction(item)
            if item== "Undo": action.setShortcut('Ctrl+Z')
            elif item== "Save": action.setShortcut('Ctrl+S')
            action.triggered.connect(self.task_master)

    def task_master(self):     #menü iteminin ismi olarak gelen string 
                                    #burada hangi işi yapması gerekiyorsa ilgili
                                    #class çağırılır ve iş gerçekleştirilir.   
        name= self.sender().text()

        if name== "Open":
            self.all_pictures= self.all_pictures[:len(self.all_pictures)-(self.current_image_index+1)]
            self.all_pictures.append(Tasks.File.Open(self, self.pic_label))
            self.current_image_index+=1            

        elif name== "Save":
            if(self.all_pictures):
                img= self.all_pictures[-1]
                r,g,b= cv2.split(img)
            
                print("Save: ")
                str= input("File name with extension: ")
                cv2.imwrite(str, cv2.merge((b,g,r)))
                os.system("cls")

        elif name== "Undo":
            if len(self.all_pictures) > 1:
                self.all_pictures.pop()
                img= self.all_pictures[-1]
                pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
                Tasks.File.refreshScreen(self.pic_label, pixmap)   
                
        elif name== "Redo":
            if self.current_image_index< len(self.all_pictures)-1:
                pass

        elif name== "Gray Level Transformation":
            if(self.all_pictures):
                img= self.all_pictures[-1]
                gray= Tasks.Preprocessing.grayLevelTransformation(img)
                self.all_pictures.append(gray)
                pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
                Tasks.File.refreshScreen(self.pic_label, pixmap)            
            
        elif name== "Resizing":
            
            img= self.all_pictures[-1]
            print("Resizing: ")
            print("Dimensions: {}*{} ".format(len(img), len(img[0])))
            width= int(input("width: "))
            height= int(input("height: "))
            

            resized= Tasks.Preprocessing.Resizing(img, height, width).astype('uint8')
            self.all_pictures.append(resized)
            pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
            Tasks.File.refreshScreen(self.pic_label, pixmap)
            os.system('cls')

        elif name== "Zoom":
            if(self.all_pictures):
                img= self.all_pictures[-1]

                print("Zoom: ")
                print("Dimensions: {}*{} ".format(len(img), len(img[0])))
                
                zoom_fact= float(input("Zoom factor: "))
                r= int(len(img)*zoom_fact)
                c= int(len(img[0])*zoom_fact) 
                zoomed= Tasks.Preprocessing.Resizing(img, r, c).astype('uint8')
                self.all_pictures.append(zoomed)
                pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
                Tasks.File.refreshScreen(self.pic_label, pixmap)
                os.system('cls')

        elif name== "Crop":
            if(self.all_pictures):
                img= self.all_pictures[-1]

                print("Crop: ")
                print("Dimensions: {}*{} ".format(len(img), len(img[0])))
                top= int(input("top: "))
                left= int(input("left: "))

                r= int(input("first dimension: "))
                c= int(input("second dimension: "))
                
                
                cropped= Tasks.Preprocessing.Crop(img,top, left, r, c)
                self.all_pictures.append(cropped)
                pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
                Tasks.File.refreshScreen(self.pic_label, pixmap)
                os.system('cls')

        elif name== "Histogram":
            if(self.all_pictures):
                img= self.all_pictures[-1]
                Tasks.Preprocessing.Histogram(img)
                #imshow penceresinde çıkacak histogram
                cv2.waitKey(0)

        elif name== "Mean Filter":
            if(self.all_pictures):
                img= self.all_pictures[-1]
                mean= Tasks.Filtering.meanFilter(img,3)
                self.all_pictures.append(mean)
                pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
                Tasks.File.refreshScreen(self.pic_label, pixmap)

        elif name== "Median Filter":
            if(self.all_pictures):
                img= self.all_pictures[-1]
                median= Tasks.Filtering.medianFilter(img,3)
                self.all_pictures.append(median)
                pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
                Tasks.File.refreshScreen(self.pic_label, pixmap)
               
        elif name== "Sharpen":
            if(self.all_pictures):
                img= self.all_pictures[-1]
                sharpen= Tasks.Filtering.sharpenFilter(img,3)
                self.all_pictures.append(sharpen)
                pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
                Tasks.File.refreshScreen(self.pic_label, pixmap)

        elif name== "Laplace Filter":
            if(self.all_pictures):
                img= self.all_pictures[-1]
                laplace= Tasks.Filtering.laplacianFilter(img,3)
                self.all_pictures.append(laplace)
                pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
                Tasks.File.refreshScreen(self.pic_label, pixmap)
        
        elif name== "Finding Edges":
            if(self.all_pictures):
                img= self.all_pictures[-1]
                find_edges= (Tasks.Filtering.findingEdges(img,3)*255).astype('uint8')
                self.all_pictures.append(find_edges)
                pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
                Tasks.File.refreshScreen(self.pic_label, pixmap)

        elif name== "Custom":
            pass

        elif name== "Dilation":
            if(self.all_pictures):
                img= self.all_pictures[-1]
                img= Tasks.Preprocessing.grayLevelTransformation(img)
                img= Tasks.Segmentation.Thresholding(img,128) 
                dilated= Tasks.Morphological_Operations.Dilation(img,3)
                self.all_pictures.append(dilated)
                pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
                Tasks.File.refreshScreen(self.pic_label, pixmap)

        elif name== "Erosion":
            if(self.all_pictures):
                img= self.all_pictures[-1]
                img= Tasks.Preprocessing.grayLevelTransformation(img)
                img= Tasks.Segmentation.Thresholding(img,128) 
                erased= Tasks.Morphological_Operations.Erosion(img,3)           
                self.all_pictures.append(erased)
                pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
                Tasks.File.refreshScreen(self.pic_label, pixmap)
            
        elif name== "Opening":
            if(self.all_pictures):
                img= self.all_pictures[-1]
                img= Tasks.Preprocessing.grayLevelTransformation(img)
                img= Tasks.Segmentation.Thresholding(img,128) 
                opened= Tasks.Morphological_Operations.Erosion(img,3)           
                opened= Tasks.Morphological_Operations.Dilation(opened,3)
                self.all_pictures.append(opened)
                pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
                Tasks.File.refreshScreen(self.pic_label, pixmap)
            
        elif name== "Closing":
            if(self.all_pictures):
                img= self.all_pictures[-1]
                img= Tasks.Preprocessing.grayLevelTransformation(img)
                img= Tasks.Segmentation.Thresholding(img,128) 
                closed= Tasks.Morphological_Operations.Dilation(img,3)           
                closed= Tasks.Morphological_Operations.Erosion(closed,3)
                self.all_pictures.append(closed)
                pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
                Tasks.File.refreshScreen(self.pic_label, pixmap)

        elif name== "Thresholding":

            if(self.all_pictures):
                img= self.all_pictures[-1]
                print("Thresholding: ")
                value= int(input("Threshhold value: "))        
                thresholded= Tasks.Segmentation.Thresholding(img,value)
                self.all_pictures.append(thresholded)
                pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
                Tasks.File.refreshScreen(self.pic_label, pixmap)
                os.system('cls')

        elif name== "Otsu Thresholding":
            if(self.all_pictures):
                img= self.all_pictures[-1]         
                thresholded= Tasks.Segmentation.OtsuThresholding(img)
                self.all_pictures.append(thresholded)
                pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
                Tasks.File.refreshScreen(self.pic_label, pixmap)

        elif name== "K- Means":
            if(self.all_pictures):
                img= self.all_pictures[-1]
                print("K- Means: ")
                center_count= int(input("Center count: "))
                clustered= Tasks.Segmentation.kMeans(img, center_count)
                self.all_pictures.append(clustered)
                pixmap= Tasks.File.getPixMap(self.all_pictures[-1])
                Tasks.File.refreshScreen(self.pic_label, pixmap)
                os.system('cls')
        
        

        

        
        
            
            


        
        


        

        
        
                        
        
        
        

