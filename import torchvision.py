import torchvision
import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split



#create a csv file to contain all file and class
folder_path = r'C:\Users\15786\Desktop\data'
folders = [f for f in os.listdir(folder_path)]
data = []
base_path = r'C:\Users\15786\Desktop\data'
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                data.append([file, folder])
df = pd.DataFrame(data, columns=['File', 'Class'])
df.to_csv(r'C:\Users\15786\Desktop\data\labels.csv',index=False)




#traun-test split
base_path = r'C:\Users\15786\Desktop\data'
train_dir = os.path.join(base_path, 'train')
val_dir = os.path.join(base_path, 'val')
test_dir = os.path.join(base_path, 'test')

for directory in [train_dir, val_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)

image_files = []
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    for file in os.listdir(folder_path):
            image_files.append(os.path.join(folder_path, file))

train_files, test_files = train_test_split(image_files, test_size=0.15, random_state=42)  # 80-20 split for train-test
train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42)  # Split remaining 80% into 60% train, 20% val

def copy_files(files, target_dir):
    for file in files:
        shutil.copy(file, target_dir)

copy_files(train_files, train_dir)
copy_files(val_files, val_dir)
copy_files(test_files, test_dir)


#generating label csv from train, test and val
folders_2 = ['train','test','val']
for folder in folders_2:
    data = []
    folder_path = os.path.join(base_path, folder)
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                data.append([file])
    subdf = pd.DataFrame(data, columns=['File'])
    result_df = pd.merge(subdf, df, on='File', how='inner')
    output_name = os.path.join(folder_path, 'labels.csv')
    result_df.to_csv(output_name,index=False)


