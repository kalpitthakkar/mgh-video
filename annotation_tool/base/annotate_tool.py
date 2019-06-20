#!/usr/bin/env python

import cv2
import numpy as np

from time import sleep

import pandas as pd
import yaml

from tkinter import *
from tkinter import ttk
import os
import glob
from configparser import ConfigParser

from .shapes import Rectangle, Circle
from .annotations import Annotation
from .events import EventHandler

class AnnotationTool(object):
    def __init__(self, data, data_type,
            with_annots, annots_file_ext,
            output_dir, yaml_config):
        
        self.data = data
        self.data_type = data_type
        self.annots_file_ext = annots_file_ext
        self.output_dir = output_dir
        self.with_annots = with_annots
        self.modes = ['VID_ANNOT_SCRATCH', 'VID_ANNOT', 'SEQ_ANNOT_SCRATCH', 'SEQ_ANNOT']

        self.annotObj = Annotation
        self.event = EventHandler()

        if self.data_type == 'Video':
            if self.with_annots:
                self.current_mode = self.modes[1]
            else:
                self.current_mode = self.modes[0]
        elif self.data_type == 'ImageSeq':
            if self.with_annots:
                self.current_mode = self.modes[3]
            else:
                self.current_mode = self.modes[2]

        with open(yaml_config, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.controls_text = ""
        for k, v in self.config['controls'].items():
            self.controls_text += ("%s: %s," % (k, v))
        self.controls_text = self.controls_text[:-1]

        self.parts = [x.strip() for x  in self.config['parts'].split(',')]
        self.attention_radius = int(self.config['attention_radius'])

    def initAnnotations(self, parts, radius, annots, player_wname, playerwidth,
            playerheight, colorDict):
       
        self.annotObj.wname = player_wname
        self.annotObj.parts_df = annots

        self.annotObj.keepWithin.x = 0
        self.annotObj.keepWithin.y = 0
        self.annotObj.keepWithin.width = playerwidth
        self.annotObj.keepWithin.height = playerheight

        self.annotObj.colorDict = colorDict
    
        for p in parts:
            self.annotObj(p)
            self.annotObj.parts[p].x_center = 0
            self.annotObj.parts[p].y_center = 0
            self.annotObj.parts[p].radius = radius
            self.annotObj.active = True
        
    def dragCircle(self, event, x, y, flags, annotObj):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.event.trigger('pressMouseButton')(x, y, annotObj)
        if event == cv2.EVENT_LBUTTONUP:
            self.event.trigger('releaseMouseButton')(x, y, annotObj)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.event.trigger('mouseDoubleClick')(x, y, annotObj)
