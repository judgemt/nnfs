import math

softmax_outputs = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.2, 0.9, 0.08]]
target_output = [1, 0, 0]

class_targets = [0, 1, 1]

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] + 
         math.log(softmax_output[2])*target_output[2])

print(loss)
loss = -math.log(softmax_output[0])
print(loss)

print(-math.log(0.7))

