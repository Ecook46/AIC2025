import pandas as pd
import os

# Processing the label text file into yolo11 annotation format
def processLabels(filepath,output):
    
    # column names from txt file
    columns = ['file','class','x1','y1','x2','y2']

    # load the txt file
    data = pd.read_csv(filepath, delimiter=' ', header=None, names=columns)

    # Preprocess the data by finding the image center, length, and width
    data['center_x'] = (data['x1'] + data['x2']) / 2
    data['center_y'] = (data['y1'] + data['y2']) / 2
    data['width'] = data['x2'] - data['x1']
    data['height'] = data['y2'] - data['y1']

    # Normalize the values by dividing by 640 (images are 640x640)
    data['center_x'] /= 640
    data['center_y'] /= 640
    data['width'] /= 640
    data['height'] /= 640
    
    # dropping extra columns
    data = data.drop(columns=['x1', 'y1', 'x2', 'y2'])

    # grouping by the file name
    groupedData = data.groupby("file")

    # exporting a txt file for each image
    for fileName, group in groupedData:
        # Prepare the file path
        file_path = os.path.join(output, f"{fileName}.txt")
    
        # Write the content of the other columns to the text file
        with open(file_path, 'a') as f:
            for _, row in group.iterrows():
                # Combineing contents of the other columns
                line = ' '.join(map(str, row[1:]))
                f.write(line + '\n')

# training directories
train = "trainingData/train/train_labels.txt"
trainOut = "trainingData/train/labels"

# validation directories
val = "trainingData/val/val_labels.txt"
valOut = "trainingData/val/labels"

# process
processLabels(val,valOut)
processLabels(train,trainOut)