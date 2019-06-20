import cv2

from .shapes import Rectangle, Circle
from .annotations import Annotation

class EventHandler(object):
    def trigger(self, ttype):
        return getattr(self, ttype)

    def pointInCircle(self, x, y, rx, ry, R):
        if((x - rx) ** 2) + ((y - ry) ** 2) < R ** 2:
            return True
        else:
            return False
        
    def updateAnnots(self, annotObj, frame_n, image):
        parts = annotObj.parts.keys()
        annot_df = annotObj.parts_df[annotObj.parts_df.frame_n == frame_n][parts]
        if annot_df.empty:
            return
        
        annotObj.image = image
        annotObj.frame_n = frame_n
        for part in annot_df:
            annotObj.parts[part].x_center, annotObj.parts[part].y_center = annot_df[part].values[0].split('-')
            #print(annot_df[joint].values[0].split('-'))

        self.clear_canvas_draw(annotObj)

    def clear_canvas_draw(self, annotObj):
        temp = annotObj.image.copy()
    
        for p in annotObj.parts:
            part = annotObj.parts[p]
            if part.x_center == 0:
                return

            x, y, r = int(part.x_center), int(part.y_center), int(part.radius)
            cv2.circle(temp, (x, y), r, annotObj.colorDict[p], -1)
            if part.focus:
                cv2.circle(temp, (x, y), r, (255, 255, 255), 2)

        cv2.imshow(annotObj.wname, temp)

    def disableResizeButtons(self, annotObj):
        annotObj.hold = False
    
    def releaseMouseButton(self, x, y, annotObj):
        if annotObj.selectedPart:
            for p in annotObj.parts:
                part = annotObj.parts[p]

                part.x_center = x
                part.y_center = y
                annotObj.parts_df.loc[annotObj.parts_df['frame_n'] == annotObj.frame_n, part.label] = str(part.x_center) + '-' + str(part.y_center)

            annotObj.selectedPart.drag = False
            self.disableResizeButtons(annotObj)
            annotObj.selectedPart.hold = False
            annotObj.selectedPart.active = False
            annotObj.selectedPart = None
            self.clear_canvas_draw(annotObj)

    def pressMouseButton(self, x, y, annotObj):
        if annotObj.selectedPart:
            return
        else:
            for p in annotObj.parts:
                part = annotObj.parts[p]

                part.x_center = x
                part.y_center = y
                annotObj.selectedPart = part
                annotObj.selectedPart.active = True
                annotObj.parts_df.loc[annotObj.parts_df['frame_n'] == annotObj.frame_n, part.label] = str(part.x_center) + '-' + str(part.y_center)

            self.clear_canvas_draw(annotObj)

    def mouseDoubleClick(self, x, y, annotObj):
        for p in annotObj.parts:
            part = annotObj.parts[p]

            if part.x_center == 0:
                return
            
            if self.pointInCircle(x, y, int(part.x_center), int(part.y_center), int(part.radius)):
                part.focus = not part.focus
                print(p + ' - Focus ' + str(part.focus))
            else:
                part.focus = False
                print(p + ' - Un-Focus ' + str(part.focus))

    def keyboardMoveMarker(self, x, y, annotObj):
        for p in annotObj.parts:
            part = annotObj.parts[p]
            if part.focus:
                annotObj.selectedPart = part

        if annotObj.selectedPart:
            part = annotObj.selectedPart
            part.x_center = x
            part.y_center = y

            if part.x_center < annotObj.keepWithin.x:
                part.x_center = annotObj.keepWithin.x
            if part.y_center < annotObj.keepWithin.y:
                part.y_center = annotObj.keepWithin.y

            if (part.x_center + part.radius) > (annotObj.keepWithin.x + annotObj.keepWithin.width - 1):
                part.x_center = annotObj.keepWithin.x + annotObj.keepWithin.width - 1 - part.radius
            if (part.y_center + part.radius) > (annotObj.keepWithin.y + annotObj.keepWithin.height - 1):
                part.y_center = annotObj.keepWithin.y + annotObj.keepWithin.height - 1 - part.radius

            annotObj.parts_df.loc[annotObj.parts_df['frame_n'] == annotObj.frame_n, part.label] = str(part.x_center) + '-' + str(part.y_center)

            self.clear_canvas_draw(annotObj)
