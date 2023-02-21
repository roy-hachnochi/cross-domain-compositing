"""Split partially augmented sofa object into training instances and testing instances."""
import os
from sklearn.model_selection import train_test_split

instance_path = "./SVR/data/SDF_v1"  # folders containing GT SDFs of ShapeNet data
cat_id = "04256520"  # category id, this is sofa

# For all models with GT SDFs, conduct train test split
all_instances = os.listdir(os.path.join(instance_path, cat_id))
train, test = train_test_split(all_instances)

# Save the train test split results in the desingated folder
save_path = "./SVR/data/train_test_split"
os.makedirs(save_path,exist_ok=True)
training_lst = os.path.join(save_path, cat_id+'_train.lst')
test_lst = os.path.join(save_path, cat_id+"_test.lst")
with open(training_lst, 'w') as filehandle:
    for instance_id in train:
        filehandle.writelines(instance_id + "\n")
with open(test_lst, 'w') as filehandle:
    for instance_id in test:
        filehandle.writelines(instance_id + "\n")