import numpy as np

from annotate_gui import AnnotationGUI

if __name__ == '__main__':
    data_dir = '/home/kalpitthakkar/g17_mount/Kalpit/bootstrap/4'
    output_dir = '/home/kalpitthakkar/g17_mount/Kalpit/bootstrap/4'
    gui = AnnotationGUI(data_dir=data_dir, output_dir=output_dir,
            data_ext='avi', with_annots=True,
            annots_file_ext='csv', yaml_config='mgh.yaml')

    root = gui.build_gui()
    root.mainloop()
