import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

init_data = [[0.9,0.3,0.8,0.2],[0.5,0.4,0.7,0.8],[0.1,0.2,0.3,0.4]]
print(len(init_data))
data = {"x": [], "y": [], "cluster": [], "student": []}
k = int(math.sqrt(len(init_data[0])))
for now in range(len(init_data)):
    vectors_set = init_data[now]
    vectors = tf.constant(vectors_set)
    centroides = tf.Variable(tf.slice(tf.nn.top_k(tf.random_shuffle(vectors),len(vectors_set),True)[0],[0],[k]))
    expanded_vectors = tf.expand_dims(vectors, 0)
    expanded_centroides = tf.expand_dims(centroides, 1)

    # 할당 단계 : 유클리드 제곱거리 사용
    diff = tf.sub(expanded_vectors, expanded_centroides)
    sqr = tf.square(diff) # 제곱
    assignments = tf.argmin(sqr,0) # 최소값 인덱스

    # # 업데이트 : 새로운 중심 계산
    means = tf.concat(0,
                    [tf.reduce_mean(
                        tf.gather(vectors,
                                    tf.reshape(
                                        tf.where(tf.equal(assignments, c))
                                        ,[1, -1])
                                    )
                        , reduction_indices=[1]) for c in range(k)])
    update_centroides = tf.assign(centroides, means)
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)
    for step in range(400):
        sess.run(update_centroides)
        assignment_values = sess.run(assignments)
    for i in range(len(assignment_values)):
        data["x"].append(vectors_set[i])
        data["student"].append(i)
        data["y"].append(now)
        data["cluster"].append(assignment_values[i])
    
df = pd.DataFrame(data)
sns.lmplot("x","y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()

student = data["student"]
print(student)
unit = data["y"]
print(unit)
cluster  = data["cluster"]
matrix = [[[]]*k for i in range(len(init_data))]
for i in range(len(student)):
    matrix[unit[i]][cluster[i]].append(student[i])
print(matrix)
