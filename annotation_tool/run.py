import numpy as np

from annotate_gui import AnnotationGUI

if __name__ == '__main__':
    data_dir = '/media/kalpitthakkar/My Book1/pose_new_subjects_06132019/BW45'
    output_dir = '/media/kalpitthakkar/My Book1/pose_new_subjects_06132019/BW45'
    gui = AnnotationGUI(data_dir=data_dir, output_dir=output_dir,
            data_ext='avi', with_annots=True,
            annots_file_ext='csv', yaml_config='mgh.yaml')

    root = gui.build_gui()
    root.mainloop()
