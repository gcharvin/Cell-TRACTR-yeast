# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 22:12:15 2022

@author: 17742
"""
from pathlib import Path
import cv2
from tkinter import Label, Tk
import numpy as np
from PIL import ImageTk, Image
from skimage.measure import label
import re


# datapath = Path(r'C:\Users\17742\Documents\DeepLearning\DeLTA_AgarPads\Evaluation\nathan\16bit_data_')
# datapath = Path(r'C:\Users\17742\Downloads\2022-04-21_TrainingSet5')
datapath = Path(r'C:\Users\17742\Documents\DeepLearning\object_detection\data\raw_data_ordered')
datapath = Path(r'C:\Users\17742\Documents\DeepLearning\object_detection\data\raw_data_ordered_20221015')
# datapath = Path(r'C:\Users\17742\Downloads\save_16bit_moma')

root = Tk()
root.title('Image')
root.geometry("450x300")
target_size = (256,256)
class gui():

    def __init__(
        self,
        datapath: str,
        target_size: tuple = (256, 32),
        ):
        
        self.target_size = target_size
        self.datapath = datapath
        self.roi_nb = 0
        self.pos_nb = 0
        self.training_set_nb = 0
        self.reset_img_path()
        self.text_frame = None
        self.my_label = None
        self.reset = True
        
        print('Click on a cell in frame t-1')
        
        # Display the first image from training_set_nb specified
        self.init_image()
        self.display_images()
                
        # Creat functionality from the keyboard to navigate left, right,
        # save images, click cells live/dead or turn chamber live/dead.
        root.bind('<Left>', self.leftKey)
        root.bind('<Right>', self.rightKey)
        root.bind('<Up>', self.upKey)
        root.bind('<Down>', self.downKey)
        root.bind('<Key>', self.key_press)
        root.bind("<Button 1>", self.getorigin)
        root.title('Live / Dead GUI')

        root.mainloop()

    def init_image(self):
        '''
        Loads the images

        '''
        self.inputs = cv2.imread(str(self.datapath / 'inputs' / (self.imgfps[self.img_nb_save].name)), cv2.IMREAD_ANYDEPTH)
        self.outputs = cv2.imread(str(self.datapath / 'outputs' / (self.imgfps[self.img_nb_save].name)), cv2.IMREAD_ANYDEPTH)
        self.seg = label(self.outputs > 0)
        
        self.previmg = cv2.imread(str(self.datapath / 'previmg' / (self.imgfps[self.img_nb_save].name)))
        self.img = cv2.imread(str(self.datapath / 'img' / (self.imgfps[self.img_nb_save].name)))
        
        self.previmg_color = self.previmg.copy()
        self.img_color = self.img.copy()
        
        self.num_cells = set(list(np.unique(self.inputs)[1:]) + list(np.unique(self.outputs)[1:]))
        np.random.seed(1)

        self.colors = [tuple((255*np.random.random(3)).astype(int)) for _ in range(len(self.num_cells))]
        self.colors_dict = {}

        self.alpha = 0.3
        self.erase = False
            
    def display_images(self):
        '''
        Displays the current image / segmentation in the GUI

        # '''
        cells_inputs = np.unique(self.inputs)[1:]
        
        if len(cells_inputs) > 0:
            for idx,cell_inputs in enumerate(cells_inputs):
                
                y,x = np.where(self.inputs == cell_inputs)
                y,x = int(np.median(y)), int(np.median(x))
                            
                self.previmg_color = cv2.putText(
                    self.previmg_color,str(cell_inputs),
                    org=(np.max((0,x-3)),y+5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 0.6,
                    color = (0,0,0),
                    thickness = 1)
                
                mask = np.zeros(self.previmg_color.shape)
                mask[self.inputs == cell_inputs] = self.colors[idx]
                self.colors_dict[str(cell_inputs)] = self.colors[idx]
                self.previmg_color[self.inputs==cell_inputs] =  self.previmg_color[self.inputs==cell_inputs] * (1-self.alpha) + mask[self.inputs==cell_inputs] * self.alpha

        cells_outputs = np.unique(self.outputs)[1:]
        
        if len(cells_outputs) > 0:
            for cell_outputs in cells_outputs:
                
                mask_label = label(self.outputs == cell_outputs)
                mask_cells = np.unique(mask_label)[1:]

                for mask_cell in mask_cells:                
                    y,x = np.where(mask_label == mask_cell)
                    y,x = int(np.median(y)), int(np.median(x))
                                
                    self.img_color = cv2.putText(
                        self.img_color,str(cell_outputs),
                        org=(np.max((0,x-3)),y+5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 0.6,
                        color = (0,0,0),
                        thickness = 1)
            
                    mask = np.zeros(self.img_color.shape)
                    if str(cell_outputs) in self.colors_dict:
                        color = self.colors_dict[str(cell_outputs)]
                    else:
                        idx += 1
                        color = self.colors[idx]
                    mask[mask_label == mask_cell] = color
                    self.img_color[mask_label == mask_cell] =  self.img_color[mask_label == mask_cell] * (1-self.alpha) + mask[mask_label == mask_cell] * self.alpha
        
        if self.text_frame is not None:
            self.text_frame.grid_forget()
        self.text_frame = Label(root, text=self.imgfps[self.img_nb_save].stem)
        self.text_frame.config(font=('Courier',15))
        self.text_frame.grid(row=1,column=1)

        self.GUI_color = np.concatenate((self.previmg_color,self.img_color,self.previmg,self.img),axis=1)
        signal_len = 5
        signal = np.zeros((signal_len,self.GUI_color.shape[1],3))
        
        if self.reset:
            signal[-signal_len:,:self.previmg_color.shape[1]] = np.array([0,255,0])
        else:
            signal[-signal_len:,self.previmg_color.shape[1]:] = np.array([0,255,0])
            
        self.GUI_color = np.concatenate((self.GUI_color,signal),axis=0)
        
        # Load the iamge and colored segmentation and display on the GUI
        # for index, img in enumerate([self.GUI_colort0, self.GUI_colort1]):
        self.GUI_color = Image.fromarray(np.uint8(self.GUI_color))
        img_PIL = ImageTk.PhotoImage(self.GUI_color)
        if self.my_label is not None:
            self.my_label.grid_forget()
        self.my_label = Label(image=img_PIL)
        self.my_label.photo = img_PIL
        self.my_label.grid(row=2, column=1)        
        
    def reset_img_path(self):
        self.imgfps = list(sorted((self.datapath / 'img').glob('*.png')))
        self.meta_fps = [np.array([list(map(int,re.findall('\d+',i.stem)))[-4] if len(list(map(int,re.findall('\d+',i.stem)))) > 3 else -1, list(map(int,re.findall('\d+',i.stem)))[-3], list(map(int,re.findall('\d+',i.stem)))[-2], list(map(int,re.findall('\d+',i.stem)))[-1], idx]) for idx,i in enumerate(self.imgfps)]
        self.meta_fps = np.stack(self.meta_fps)
        
        self.unique_training_sets = np.unique(self.meta_fps[:,0])
            
        self.rois = []
        self.positions = []
        
        training_set_fps = self.meta_fps[self.meta_fps[:,0] == self.unique_training_sets[self.training_set_nb]]

        self.training_set_imgnbs = training_set_fps[:,-1]
        self.img_nb_save = self.training_set_imgnbs[0]
        positions = np.unique(training_set_fps[:,1])
        for pos in positions:
            pos_fps = training_set_fps[training_set_fps[:,1] == pos]
            self.positions.append(pos_fps[0,-1])
            rois = np.unique(pos_fps[:,2])
            for roi in rois:
                roi_fps = pos_fps[pos_fps[:,2] == roi]
                self.rois.append(roi_fps[0,-1])
                
        
        
    def leftKey(self, event):
        '''
        Pressing the left key will look at the previous image''

        '''
        if self.img_nb_save - 1 < min(self.training_set_imgnbs):
            pass
        else:
            self.my_label.config(image='')
            self.reset=True
            self.img_nb_save -= 1
            self.init_image()
            self.display_images()

    def rightKey(self, event):
        '''
        Pressing the right key will look at the next image''

        '''
        if self.img_nb_save + 1 >  max(self.training_set_imgnbs):
            pass
        else:
            self.my_label.config(image='')
            self.reset=True
            self.img_nb_save += 1
            self.init_image()
            self.display_images()
            
    def upKey(self, event):
        '''
        Pressing the right key will look at the next image''

        '''
        if self.img_nb_save + 10 >  max(self.training_set_imgnbs):
            pass
        else:
            self.my_label.config(image='')
            self.reset=True
            self.img_nb_save += 10
            self.init_image()
            self.display_images()
            
    def downKey(self, event):
        '''
        Pressing the right key will look at the next image''

        '''
        if self.img_nb_save - 10 < min(self.training_set_imgnbs):
            pass
        else:
            self.my_label.config(image='')
            self.reset=True
            self.img_nb_save -= 10
            self.init_image()
            self.display_images()
                        
    def getorigin(self, eventorigin):
        '''
        Gets the location of the cursor.

        Parameters
        ----------
        eventorigin : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Gets location of cursor
        x = int(eventorigin.x)
        y = int(eventorigin.y)

        if self.erase:
            if y < self.seg.shape[0] and x < self.seg.shape[1]:
                cellnb = self.inputs[y,x]
                
                if cellnb == 0:
                    print('You must click on a cell')
                else:
                    self.inputs[self.inputs==cellnb] = 0
                    self.save()
            elif y < self.seg.shape[0] and x < self.seg.shape[1]*2:
                cellnb = self.outputs[y,x-self.seg.shape[1]]
                if cellnb == 0:
                    print('You must click on a cell')
                else:
                    self.outputs[self.outputs==cellnb] = 0
                    self.seg[self.outputs==cellnb] = 0
                    
                    self.save()
            self.erase=False
                
        else:
            if self.reset == True:
                
                if y > self.seg.shape[0] or x > self.seg.shape[1]:
                    print('Click on a cell in first frame')
                else:
                    self.cellnb = self.inputs[y,x]
                       
                    if self.cellnb == 0:
                        print('You missed the cell')
                        self.reset = True
                            
                    else:
                        self.color = self.previmg_color[y,x]
                        self.reset=False
                        print('Click on a cell in frame t')
     
            else:
                if y > self.seg.shape[0] or x < self.seg.shape[1] or x > self.seg.shape[1]*2:
                    print('Click on cell in second frame')
                else:
                    x = x - self.seg.shape[1]
                    self.cellnb_output = self.seg[y,x]
                    
                    if self.cellnb_output == 0:
                        print('You missed the cell')
                        self.reset=False
                          
                    else:
                               
                        self.outputs[self.seg==self.cellnb_output] = self.cellnb
                        mask = np.zeros((self.img_color.shape))
                        mask[self.seg==self.cellnb_output] = self.color
                        self.img_color[self.seg==self.cellnb_output] = self.img_color[self.seg==self.cellnb_output] * (1-self.alpha) + mask[self.seg==self.cellnb_output] *self.alpha
                        self.reset=True
                        print('Click on a cell in frame t-1')
                    
                    self.save()
        self.init_image()
        self.display_images()

    def key_press(self, event):
        
        key = event.char
        
        if key == 'q':
            print(key)
            self.reset = True
            self.display_images()
            
        elif key == 'd':
            print('Delete file')
            for folder_path in self.datapath.iterdir():
                (folder_path / self.imgfps[self.img_nb_save].name).unlink()
                
            self.reset_img_path()
            self.reset = True
            self.init_image()
            self.display_images()
            
        elif key == 'r':
            if self.roi_nb + 1 != len(self.rois):
                self.roi_nb += 1
            else:
                self.roi_nb = 0
                
            self.img_nb_save = self.rois[self.roi_nb]
            self.reset=True
            self.init_image()
            self.display_images()
               
        elif key == 't':
            if self.training_set_nb + 1 != len(self.unique_training_sets):
                self.training_set_nb += 1
            else:
                self.training_set_nb = 0
                
            self.pos_nb = 0
            self.roi_nb = 0
            self.reset_img_path()
            self.img_nb_save = self.rois[self.roi_nb]
            self.reset=True
            self.init_image()
            self.display_images()
            
        elif key == 'p':
            if self.pos_nb + 1 != len(self.positions):
                self.pos_nb += 1

            else:
                self.pos_nb = 0
                
            self.img_nb_save = self.positions[self.pos_nb]
            self.roi_nb = self.rois.index(self.img_nb_save)
            self.reset=True
            self.init_image()
            self.display_images()
            
        elif key == 'e':
            print(key)
            self.erase = True
        
        
    def save(self):
        '''
        This function saves the results in the results folder. 
        It will also save the colored results if a color folder was specified.

        Returns
        -------
        None.

        '''
        
        cv2.imwrite(str(self.datapath / 'color' / (self.imgfps[self.img_nb_save].name)),np.concatenate((self.previmg_color,self.img_color),axis=1))
        cv2.imwrite(str(self.datapath / 'inputs' / self.imgfps[self.img_nb_save].name),self.inputs)
        cv2.imwrite(str(self.datapath / 'outputs' / self.imgfps[self.img_nb_save].name),self.outputs)
        cv2.imwrite(str(self.datapath / 'segall' / self.imgfps[self.img_nb_save].name),self.seg)
        
        print('save')
              
gui(
    datapath = datapath,
    target_size = target_size
    )