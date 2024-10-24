import datetime
import pathlib
from typing import Optional, List, Dict, Union

import numpy as np
import os
import ntpath
import time

from matplotlib import pyplot as plt
import matplotlib.patches as plt_patch
from monai.data import MetaTensor

from . import util
# from . import html
#import matplotlib
#import matplotlib.pyplot as plt
#plt.switch_backend('agg')

#matplotlib.use('agg')

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = self.opt.display_single_pane_ncols
            if ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    #images.append(image_numpy.transpose([2, 0, 1]))
                    images.append(image_numpy.transpose([2, 0, 1])/1500.)
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            #webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            #for n in range(epoch, 0, -1):
            #    webpage.add_header('epoch [%d]' % n)
            #    ims = []
            #    txts = []
            #    links = []

            #    for label, image_numpy in visuals.items():
            #        img_path = 'epoch%.3d_%s.png' % (n, label)
            #        ims.append(img_path)
            #        txts.append(label)
            #        links.append(img_path)
            #    webpage.add_images(ims, txts, links, width=self.win_size)
            #webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    def save_current_errors(self, epoch, counter_ratio, opt, errors,sv_name):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])

        #print (self.plot_data['X'])

        #print (self.plot_data['Y'])

        #print (self.plot_data['legend'])

    def get_cur_plot_error(self, epoch, counter_ratio, opt, errors,sv_name):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])

        return self.plot_data
        #plt.plot(self.plot_data['X'], y[:,0], label='training loss')
        #plt.plot(x, y[:,1], label='training_rpn_class_loss')
        #plt.plot(x, y[:,2], label='training_rpn_bbox_loss')

        #self.vis.line(
        #    X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
        #    Y=np.array(self.plot_data['Y']),
        #    opts={
        #        'title': self.name + ' loss over time',
        #        'legend': self.plot_data['legend'],
        #        'xlabel': 'epoch',
        #        'ylabel': 'loss'},
        #    win=self.display_id)
        

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)


def visualize_case_slices(
        image_data: List[Union["MetaTensor", np.ndarray]],
        contour_data: List[Union["MetaTensor", np.ndarray]],
        contour_colors: List[str],
        contour_labels: List[str],
        save_folder: Optional[pathlib.Path] = None,
        additional_info: Optional[Dict[str, Union[float, int, str]]] = None,
        case_id: Optional[str] = None
) -> Optional[pathlib.Path]:
    """
    Visualize multiple image slices with contour overlays in a single figure.

    This function takes lists of image data and contour data, and visualizes
    all slices in one figure. The resulting plot shows images in columns,
    with contour overlays on each image.

    Args:
        image_data (List[Union[MetaTensor, np.ndarray]]): List of image data (MetaTensor or numpy array).
        contour_data (List[Union[MetaTensor, np.ndarray]]): List of contour data (MetaTensor or numpy array).
        contour_colors (List[str]): List of colors for each contour.
        contour_labels (List[str]): List of labels for each contour.
        save_folder (Optional[pathlib.Path]): Path to the folder where the visualization should be saved.
                                              If None, the plot is displayed but not saved.
        additional_info (Optional[Dict[str, Union[float, int, str]]]): Dictionary containing additional
                                                                       information to display in the title.
        case_id (Optional[str]): Identifier for the specific case. If provided, it will be shown in the plot
                                 and used in the filename when saving.

    Returns:
        Optional[pathlib.Path]: The path to the saved visualization if save_folder is provided, else None.

    Example:
        >>> adc = MetaTensor(...)  # 5D MetaTensor
        >>> t2 = MetaTensor(...)   # 5D MetaTensor
        >>> label = MetaTensor(...) # 5D MetaTensor
        >>> prediction = np.array(...) # 3D numpy array
        >>> visualize_patient_slices(
        ...     image_data=[adc, t2],
        ...     contour_data=[label, prediction],
        ...     contour_colors=['r', 'b'],
        ...     contour_labels=['Ground Truth', 'Prediction'],
        ...     save_folder=pathlib.Path('./output'),
        ...     additional_info={'Dice': 0.85, 'Epoch': 10},
        ...     case_id='CASE001'
        ... )
    """
    # Ensure inputs are valid
    if len(image_data) != len(contour_data):
        raise ValueError("Number of image data and contour data must match")

    if len(contour_data) != len(contour_colors) or len(contour_data) != len(contour_labels):
        raise ValueError("Number of contour data, colors, and labels must match")

    # Process image and contour data
    processed_images = []
    processed_contours = []

    for img, contour in zip(image_data, contour_data):
        if isinstance(img, MetaTensor):
            img = img[0, 0, :, :, :].cpu().numpy()
        processed_images.append(img)

        if isinstance(contour, MetaTensor):
            contour = contour[0, 0, :, :, :].cpu().numpy()
        processed_contours.append(contour)

    num_images = len(processed_images)
    num_slices = processed_images[0].shape[2]  # Assuming the last dimension is the number of slices

    # Create figure with columns for each image type
    fig, axes = plt.subplots(num_slices, num_images, figsize=(5 * num_images, 5 * num_slices))

    def create_suptitle(additional_info, case_id):
        title = "Image Slices with Contour Overlays"
        if case_id:
            title += f"\nCase ID: {case_id}"
        if additional_info:
            info_strings = [f"{key}: {value}" for key, value in additional_info.items()]
            title += f"\n{' | '.join(info_strings)}"
        return title

    fig.suptitle(create_suptitle(additional_info, case_id), fontsize=16)

    def plot_slice(ax, image_slice, contour_slices, contour_colors, slice_index, image_type):
        ax.imshow(image_slice, cmap='gray')
        for contour_slice, color in zip(contour_slices, contour_colors):
            ax.contour(contour_slice, colors=color, linewidths=0.5)
        ax.axis('off')
        ax.set_title(f'{image_type} Slice {slice_index}')

    def get_legend_elements(contour_labels, contour_colors):
        return [plt_patch.Patch(facecolor=color, edgecolor=color, label=label)
                for label, color in zip(contour_labels, contour_colors)]

    for slice_index in range(num_slices):
        for img_index, (image, contours) in enumerate(zip(processed_images, processed_contours)):
            ax = axes[slice_index, img_index] if num_images > 1 else axes[slice_index]
            image_slice = image[:, :, slice_index]
            contour_slices = [contour[:, :, slice_index] for contour in processed_contours]
            plot_slice(ax, image_slice, contour_slices, contour_colors, slice_index, f'Image {img_index + 1}')

    # Add legend to the figure
    legend_elements = get_legend_elements(contour_labels, contour_colors)
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(contour_labels), bbox_to_anchor=(0.5, 0.02))

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05, top=0.95)  # Make room for the legend and title

    # Save the visualization if save_folder is provided
    if save_folder:
        save_folder = pathlib.Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if case_id:
            file_name = f'case_{case_id}_visualization_{current_time}.png'
        else:
            file_name = f'case_visualization_{current_time}.png'

        image_path = save_folder / file_name
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved visualization to: {image_path}")
        return image_path
    else:
        plt.show()
        return None
