import numpy as np
import matplotlib.pyplot as plt


def aligned_component_plot(component):
    """
    Visualize alignment in the component.
    :param component: a groupedROI object
    """
    name2label = {}
    label2class = {}
    labels = set()
    for name in component.roi:
        end = name.rfind('/')
        begin = name[:end].rfind('/') + 1
        label = name[begin:end]
        labels.add(label)
        name2label[name] = label

    for i, label in enumerate(labels):
        label2class[label] = i

    m = len(labels)
    mz = []
    scan_begin = []
    scan_end = []
    for file, shift in zip(component.roi, component.shifts):
        for roi in component.roi[file]:
            mz.append(roi.mzmean)
            scan_begin.append(roi.scan[0] + shift)
            scan_end.append(roi.scan[1] + shift)
            y = roi.i
            x = np.linspace(roi.scan[0], roi.scan[1], len(y))
            x_shifted = np.linspace(roi.scan[0] + shift, roi.scan[1] + shift, len(y))
            label = label2class[name2label[file]]
            c = [label / m, 0.0, (m - label) / m]
            plt.subplot(121)
            plt.plot(x, y, color=c)
            plt.subplot(122)
            plt.plot(x_shifted, y, color=c)
    plt.title('mz = {:.4f}, scan = {:.2f} -{:.2f}'.format(np.mean(mz), min(scan_begin), max(scan_end)))
