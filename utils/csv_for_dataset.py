import os
import pandas
import re
import numpy

labels = ['Noodles-Pasta', 'Egg', 'Meat', 'Dessert', 'Rice', 'Vegetable-Fruit', 'Seafood', 'Dairy product', 'Bread', 'Soup', 'Fried food']
#create a csv file of each image path from the root and its label
def extract_csv_for_dataset(root, out_file):
  csv = []
  print(labels)

  for label, label_name in enumerate(labels):
    path = os.path.join(root, label_name)
    for image in os.listdir(path):
      if re.match(r"\d+.jpg", image):
        path_to_image = os.path.join(path, image)
        csv.append([path_to_image, label])

  csv = numpy.array(csv)
  df = pandas.DataFrame(csv)
  df.to_csv(os.path.join(root, out_file), index=False)
  


if __name__ == "__main__":
  extract_csv_for_dataset("./data/food11k/evaluation", "food11k.csv")