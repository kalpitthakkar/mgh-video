#!/usr/bin/env python

import cv2
import numpy as np

from time import sleep

import pandas as pd

from tkinter import *
from tkinter import ttk
import os
from configparser import ConfigParser

from base import Annotation, AnnotationTool, Rectangle, Circle, EventHandler

import matplotlib.pyplot as plt

def flick(x):
    pass

class AnnotationGUI(object):
    def __init__(self, data, output_dir,
            data_type='Video', with_annots=True, annots_file_ext='csv',
            yaml_config='ksu_mice.yaml'):

        self.left_annots = None
        self.right_annots = None
        self.data = data
        self.annotTool = AnnotationTool(
            data, data_type, with_annots, annots_file_ext,
            output_dir, yaml_config
        )
        
    def onselect(self, evt):
        # Note here that Tkinter passes an event object to onselect()
        w = evt.widget
        data = self.annotTool.data
        index = int(w.curselection()[0])
        if index == 0:
            chunk = 'left'
            v_data = np.squeeze(data[0])
        else:
            chunk = 'right'
            v_data = np.squeeze(data[1])
        self.show_video_with_annots(v_data, data[2], data[3], data[4], chunk)
    
    def cv2WindowInit(self, v_data, vname, label, frame, chunk='Left', bb_path=None):
        vname = str(vname[0].decode("utf-8"))
        self.save_path_prefix = os.path.join(self.annotTool.output_dir, vname)
        if not os.path.exists(self.save_path_prefix):
            os.makedirs(self.save_path_prefix)

        player_wname = 'Data chunk - ' + chunk
        control_wname = 'Controls'
        color_wname = 'Color mappings'
        
        cv2.destroyAllWindows()
        cv2.namedWindow(player_wname, cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow(player_wname, 400, 335)
        cv2.namedWindow(control_wname)
        cv2.moveWindow(control_wname, 400, 50)
        cv2.namedWindow(color_wname)
        cv2.moveWindow(color_wname, 400, 190)

        playerwidth = v_data[0].shape[0]
        playerheight = v_data[0].shape[1]
        if bb_path:
            self.annots = pd.read_csv(bb_path)
        elif os.path.exists(os.path.join(self.save_path_prefix, 'annots_'+chunk.lower()+'_label'+str(label[0])+'_'+str(frame[0])+'.csv')):
            self.annots = pd.read_csv(os.path.join(self.save_path_prefix, 'annots_'+chunk.lower()+'_label'+str(label[0])+'_'+str(frame[0])+'.csv')) 
        else:
            columns = ['video_path', 'frame_n']
            for p in self.annotTool.parts:
                columns.append(p)       # For each part, the annotation should be x-y
            df_dict = {k: [] for k in columns}
            num_frames = v_data.shape[0]
            for i in range(num_frames):
                df_dict['video_path'].append(vname)
                df_dict['frame_n'].append(i)
                for p in self.annotTool.parts:
                    df_dict[p].append("0-0")
            self.annots = pd.DataFrame.from_dict(df_dict)
        colorList = [[0, 0, 255]]
        colorDict = dict(zip(self.annotTool.parts, colorList))

        self.annotTool.initAnnotations(self.annotTool.parts, self.annotTool.attention_radius, self.annots,
                player_wname, playerwidth, playerheight, colorDict)
        cv2.setMouseCallback(player_wname, self.annotTool.dragCircle, self.annotTool.annotObj)
        self.controls = np.zeros((140, int(playerwidth * 2)), np.uint8)
        y0, dy = 20, 25
        for i, line in enumerate(self.annotTool.controls_text.split('\n')):
            y = y0 + i * dy
            cv2.putText(self.controls,
                        line + ' ',
                        (0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        i, x0, y = 0, 0, 20
        x = [0, 85, 270, 400, 510, 680, 800]
        self.color_map = np.zeros((40, int(playerwidth * 2), 3), np.uint8)
        self.color_map[:, :] = 255
        for this_joint in self.annotTool.parts:
            this_color = colorDict[this_joint]
            this_color = tuple(this_color)
            cv2.putText(self.color_map, this_joint, (x[i], y), cv2.FONT_HERSHEY_SIMPLEX, 1, this_color, 2)
            i += 1

        tots = len(self.annots.index)
        cv2.createTrackbar('S', player_wname, 0, int(tots) - 1, flick)
        cv2.setTrackbarPos('S', player_wname, 0)
        cv2.createTrackbar('F', player_wname, 1, 10, flick)
        frame_rate = 1
        cv2.setTrackbarPos('F', player_wname, frame_rate)
        
    def show_video_with_annots(self, v_data, label, vname, frame, chunk):
        if chunk == 'left':
            self.cv2WindowInit(v_data, vname, label, frame, 'Left')
            player_wname = 'Data chunk - Left'
        else:
            self.cv2WindowInit(v_data, vname, label, frame, 'Right')
            player_wname = 'Data chunk - Right'
        control_wname = 'Controls'
        color_wname = 'Color mappings'
        
        tots = v_data.shape[0]
        i = 0
        status = 'stay'
        while True:
            playerwidth = self.annotTool.annotObj.keepWithin.width
            playerheight = self.annotTool.annotObj.keepWithin.height
            cv2.imshow(control_wname, self.controls)
            cv2.imshow(color_wname, self.color_map)
            im = np.squeeze(v_data[i])
            if i == tots:
                i = 0
                status = 'stay'
            r = playerwidth / im.shape[1]
            dim = (int(playerwidth), int(im.shape[0] * r))
            im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow(player_wname, im)
            self.annotTool.event.updateAnnots(self.annotTool.annotObj, i, im)

            key = cv2.waitKey(10)
            status = {ord('s'): 'stay', ord('S'): 'stay',
                      ord('w'): 'play', ord('W'): 'play',
                      ord('a'): 'prev_frame', ord('A'): 'prev_frame',
                      ord('d'): 'next_frame', ord('D'): 'next_frame',
                      ord('q'): 'copy', ord('Q'): 'copy',
                      ord('z'): 'save', ord('Z'): 'save',
                      ord('c'): 'quit',
                      ord('x'): 'incorrect_num',
                      ord('i'): 'move_marker_up',
                      ord('m'): 'move_marker_down',
                      ord('j'): 'move_marker_left',
                      ord('l'): 'move_marker_right',
                      255: status,
                      -1: status,
                      27: 'exit'}[key]

            if status == 'move_marker_up':
                for joint_name in self.annotTool.annotObj.parts:
                    joint = self.annotTool.annotObj.parts[joint_name]
                    if joint.focus:
                        self.annotTool.annotObj.selectedPart = joint

                if self.annotTool.annotObj.selectedPart:
                    joint = self.annotTool.annotObj.selectedPart
                    curr_x, curr_y = int(joint.x_center), int(joint.y_center)
                    self.annotTool.event.trigger('pressMouseButton')(curr_x, curr_y, self.annotTool.annotObj)
                    self.annotTool.event.trigger('keyboardMoveMarker')(curr_x, curr_y-1, self.annotTool.annotObj)
                    self.annotTool.event.trigger('releaseMouseButton')(curr_x,curr_y-1, self.annotTool.annotObj)

                status = 'stay'
                if i % 10 == 0:
                    self.annots.to_csv(os.path.join(self.save_path_prefix, 'annots_'+chunk+'_label'+str(label[0])+'_'+str(frame[0])+'.csv'), index=False)
            if status == 'move_marker_down':
                for joint_name in self.annotTool.annotObj.parts:
                    joint = self.annotTool.annotObj.parts[joint_name]
                    if joint.focus:
                        self.annotTool.annotObj.selectedPart = joint

                if self.annotTool.annotObj.selectedPart:
                    joint = self.annotTool.annotObj.selectedPart
                    curr_x, curr_y = int(joint.x_center), int(joint.y_center)
                    self.annotTool.event.trigger('pressMouseButton')(curr_x, curr_y, self.annotTool.annotObj)
                    self.annotTool.event.trigger('keyboardMoveMarker')(curr_x, curr_y+1, self.annotTool.annotObj)
                    self.annotTool.event.trigger('releaseMouseButton')(curr_x,curr_y+1, self.annotTool.annotObj)

                status = 'stay'
                if i % 10 == 0:
                    self.annots.to_csv(os.path.join(self.save_path_prefix, 'annots_'+chunk+'_label'+str(label[0])+'_'+str(frame[0])+'.csv'), index=False)
            if status == 'move_marker_left':
                for joint_name in self.annotTool.annotObj.parts:
                    joint = self.annotTool.annotObj.parts[joint_name]
                    if joint.focus:
                        self.annotTool.annotObj.selectedPart = joint

                if self.annotTool.annotObj.selectedPart:
                    joint = self.annotTool.annotObj.selectedPart
                    curr_x, curr_y = int(joint.x_center), int(joint.y_center)
                    self.annotTool.event.trigger('pressMouseButton')(curr_x, curr_y, self.annotTool.annotObj)
                    self.annotTool.event.trigger('keyboardMoveMarker')(curr_x-1, curr_y, self.annotTool.annotObj)
                    self.annotTool.event.trigger('releaseMouseButton')(curr_x-1,curr_y, self.annotTool.annotObj)

                status = 'stay'
                if i % 10 == 0:
                    self.annots.to_csv(os.path.join(self.save_path_prefix, 'annots_'+chunk+'_label'+str(label[0])+'_'+str(frame[0])+'.csv'), index=False)
            if status == 'move_marker_right':
                for joint_name in self.annotTool.annotObj.parts:
                    joint = self.annotTool.annotObj.parts[joint_name]
                    if joint.focus:
                        self.annotTool.annotObj.selectedPart = joint

                if self.annotTool.annotObj.selectedPart:
                    joint = self.annotTool.annotObj.selectedPart
                    curr_x, curr_y = int(joint.x_center), int(joint.y_center)
                    self.annotTool.event.trigger('pressMouseButton')(curr_x, curr_y, self.annotTool.annotObj)
                    self.annotTool.event.trigger('keyboardMoveMarker')(curr_x+1, curr_y, self.annotTool.annotObj)
                    self.annotTool.event.trigger('releaseMouseButton')(curr_x+1,curr_y, self.annotTool.annotObj)

                status = 'stay'
                if i % 10 == 0:
                    self.annots.to_csv(os.path.join(self.save_path_prefix, 'annots_'+chunk+'_label'+str(label[0])+'_'+str(frame[0])+'.csv'), index=False)
            if status == 'play':
                frame_rate = cv2.getTrackbarPos('F', player_wname)
                sleep((0.1 - frame_rate / 1000.0) ** 21021)
                i += 1

                if i == tots:
                    i = 0
                cv2.setTrackbarPos('S', player_wname, i)
                continue
            if status == 'stay':
                i = cv2.getTrackbarPos('S', player_wname)
            if status == 'save':
                self.annots.to_csv(os.path.join(self.save_path_prefix, 'annots_'+chunk+'_label'+str(label[0])+'_'+str(frame[0])+'.csv'), index=False)
                print('Progress saved!')
                if chunk == 'left':
                    self.left_annots = self.annots
                else:
                    self.right_annots = self.annots
                status = 'stay'
            if status == 'quit':
                self.annots.to_csv(os.path.join(self.save_path_prefix, 'annots_'+chunk+'_label'+str(label[0])+'_'+str(frame[0])+'.csv'), index=False)
                print('Quit. Progress automatically saved!')
                if chunk == 'left':
                    self.left_annots = self.annots
                else:
                    self.right_annots = self.annots
                break
            if status == 'exit':
                self.annots.to_csv(os.path.join(self.save_path_prefix, 'annots_'+chunk+'_label'+str(label[0])+'_'+str(frame[0])+'.csv'), index=False)
                print('Save & Quit!')
                if chunk == 'left':
                    self.left_annots = self.annots
                else:
                    self.right_annots = self.annots
                break
            if status == 'prev_frame':
                i -= 1
                cv2.setTrackbarPos('S', player_wname, i)
                status = 'stay'
            if status == 'next_frame':
                i += 1
                if i == tots:
                    i = 0
                cv2.setTrackbarPos('S', player_wname, i)
                status = 'stay'
            if status == 'copy':
                if i != 0:
                    self.annots['attention_loc'].iloc[i] = self.annots['attention_loc'].iloc[i - 1]
                if i % 10 == 0:
                    self.annots.to_csv(os.path.join(self.save_path_prefix, 'annots_'+chunk+'_label'+str(label[0])+'_'+str(frame[0])+'.csv'), index=False)
                status = 'stay'
            if status == 'slow':
                frame_rate = max(frame_rate - 5, 0)
                cv2.setTrackbarPos('F', player_wname, frame_rate)
                status = 'play'
            if status == 'fast':
                frame_rate = min(100, frame_rate + 5)
                cv2.setTrackbarPos('F', player_wname, frame_rate)
                status = 'play'
            if status == 'snap':
                cv2.imwrite("./" + "Snap_" + str(i) + ".jpg", im)
                print("Snap of Frame", i, "Taken!")
                status = 'stay'
            if status == 'incorrect_num':
                # TODO: Add this function
                # debug_list = self.annotTool.event.debug(self.annots)
                # print('num_incorrect:' + str(len(debug_list)))
                status = 'stay'

        cv2.destroyWindow(player_wname)
        cv2.destroyWindow(control_wname)
        cv2.destroyWindow(color_wname)

    def build_gui(self):
        root = Tk()
        l = Listbox(root, selectmode=SINGLE, height=30, width=60)
        l.grid(column=0, row=0, sticky=(N, W, E, S))
        s = ttk.Scrollbar(root, orient=VERTICAL, command=l.yview)
        s.grid(column=1, row=0, sticky=(N, S))
        l['yscrollcommand'] = s.set
        ttk.Sizegrip().grid(column=1, row=1, sticky=(S, E))
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)
        root.geometry('350x500+50+50')
        root.title('Select Video')
        l.insert(END, 'Left Chunk')
        l.insert(END, 'Right Chunk')

        l.bind('<<ListboxSelect>>', self.onselect)
        return root
