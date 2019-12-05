import sys
import os
import torch
from tqdm import tqdm
from utils.roi import get_ROIs
from utils.models import Classifier, Integrator
from utils.matching import construct_mzregions, rt_grouping, align_component
from utils.run_utils import find_mzML, classifier_prediction, border_prediction,\
    correct_classification, border_correction, build_features
from utils.postprocess import ResultTable

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('''Run script in the following format:
    python3 run.py path_to_file delta_mz roi_minimum_points peak_minimum_points''')
        exit()
    path = sys.argv[1]
    delta_mz = float(sys.argv[2])
    required_points = int(sys.argv[3])
    peak_minimum_points = int(sys.argv[4])

    ### ROI detection ###
    # search .mzML files in directory
    files = find_mzML(path)
    print('In the directory {} found {} files'.format(path, len(files)))
    # get ROIs for every file
    ROIs = {}
    for file in files:
        ROIs[file] = get_ROIs(file, delta_mz, required_points)

    ### ROI alignment ###
    # construct mz regions
    mzregions = construct_mzregions(ROIs, delta_mz)
    # group ROIs in mz regions based on RT
    components = rt_grouping(mzregions)
    # component alignment
    aligned_components = []
    for component in components:
        aligned_components.append(align_component(component))

    ### Classification, integration and correction ###
    # load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classify = Classifier()
    classify.load_state_dict(torch.load('data/Classifier', map_location=device))
    classify.to(device)
    classify.eval()
    integrate = Integrator()
    integrate.load_state_dict(torch.load('data/Integrator', map_location=device))
    integrate.to(device)
    integrate.eval()
    # run through components
    features = []
    for component in tqdm(aligned_components):
        # predict labels and correct them
        labels = {}
        for sample, roi in zip(component.samples, component.rois):
            labels[sample] = classifier_prediction(roi, classify, device)
        correct_classification(labels)
        # predict borders and correct them
        borders = {}
        to_delete = []
        for j, (sample, roi) in enumerate(zip(component.samples, component.rois)):
            if labels[sample] == 1:
                border = border_prediction(roi, integrate, device, peak_minimum_points)
                if len(border) == 0:  # if no border were predicted
                    to_delete.append(j)
                else:
                    borders[sample] = border
            else:
                to_delete.append(j)
        if len(borders) > len(files) // 3:  # enough rois contain a peak
            component.pop(to_delete)  # delete ROIs which don't contain peaks
            # to do: add multi functionality
            uni = True
            for border in borders.values():
                if len(border) > 1:
                    uni = False
            if uni:
                border_correction(component, borders)
                features.append(build_features(component, borders))
    print('total number of features: {}'.format(len(features)))

    ### Save all features to csv file (zero filling is missing now)###
    table = ResultTable(files, features)
    table.fill_zeros(delta_mz)
    table.to_csv(os.path.join(path, 'resultTable.csv'))
