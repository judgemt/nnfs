import numpy as np
# each row is the distribution of class probs for a sample in the batch:
softmax_outputs = np.array([[0.7, 0.1, 0.2],        # sample 1
                            [0.1, 0.5, 0.4],        # sample 2 
                            [0.02, 0.9, 0.08]])     # sample 3

class_targets = [0, 1, 1]

class_list = range(len(softmax_outputs))

neg_log = -np.log(softmax_outputs[[class_list], class_targets])

average_loss = np.mean(neg_log)

print(average_loss)

# -log(0) is inf !

clipped = np.clip(average_loss, 1e-7, 1-1e-7)

# Accuracy

predictions = np.argmax(softmax_outputs, axis=1)

print("Accuracy = ", np.mean(predictions == class_targets))