import frozen_features

from sklearn.svm import SVC
import time

trainset, ft_concat_size = frozen_features.data_set_from_csv("food11ktrain.csv", 1)

X_train = []
y_train = []
for labels_tensor, images_tensor in trainset:
  X_train = X_train + images_tensor.tolist()
  y_train = y_train + labels_tensor.tolist()

print(len(X_train[0]))

testset, _ = frozen_features.data_set_from_csv("food11ktest.csv", 1)

X_test = []
y_test = []
for labels_tensor, images_tensor in testset:
  X_test = X_test + images_tensor.tolist()
  y_test = y_test + labels_tensor.tolist()




# Define the model with cost set to 0.1
model = SVC(C=0.1)

start = time.time()
# Fit the model to your training data (X_train, y_train)
model.fit(X_train, y_train)
print(f"finished building svm for: {time.time() - start}")

correct = 0 
total = 0
predictions = model.predict(X_test)

for index, pred in enumerate(predictions):
  # print("a" + str(pred))
  # print(y_test[index])

  total += 1
  if pred == y_test[index]:
    print("yay")
    correct += 1
print(correct / total)