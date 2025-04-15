import numpy as np

from numpy import inf

emitters = ["emitter_0", "emitter_1", "emitter_2", "emitter_3"]
action = [1, 2, 4, 10, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
common_data = action[:2]
specific_data = [action[i:i+3] for i in range(2, 14, 3)]
print(common_data)
print(specific_data)
for i, emitter in enumerate(emitters):
    message_data = common_data + specific_data[i]
    message = (",".join(map(str, message_data)))
    print("i:", message)
data1 = np.array([12, 10, 5])
data2 = np.array([1, 0, 0])
data = np.hstack((data1, data2))
print(data.reshape((1, 2, 3)))
