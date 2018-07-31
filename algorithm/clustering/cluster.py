import numpy as np

num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0, 0.9),np.random.normal(0.0, 0.9)])

    else:
        vectors_set.append([np.random.normal(3.0, 0.5),np.random.normal(1.0, 0.5)])

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.DataFrame({"x": [v[0] for v in vectors_set],\
                   "y": [v[1] for v in vectors_set]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
# plt.show()

import tensorflow as tf

    # 모든 데이터를 상수 텐서로 옮김
vectors = tf.constant(vectors_set)
    # 초기 단계 : 중심 k(4)개를 입력데이터에서 무작위로 선택
k = 2
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))
    # vector.get_shape()
    # centroides.get_shape()
expanded_vectors = tf.expand_dims(vectors, 0)
    # print(expanded_vectors.get_shape())
expanded_centroides = tf.expand_dims(centroides, 1)
    # print(expanded_centroides.get_shape())

    # 할당 단계 : 유클리드 제곱거리 사용
diff = tf.sub(expanded_vectors, expanded_centroides)
    # print(diff.get_shape())
sqr = tf.square(diff) # 제곱
distances = tf.reduce_sum(sqr, 2)   # x+y
    # print(distances.get_shape())
assignments = tf.argmin(distances,0) # 최소값 인덱스
    # print(assignments.get_shape())
    
# 업데이트 : 새로운 중심 계산
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

# print(sess.run(means))
for step in range(10):
    _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])

    data = {"x": [], "y": [], "cluster": []}

    for i in range(len(assignment_values)):
        data["x"].append(vectors_set[i][0])
        data["y"].append(vectors_set[i][1])
        data["cluster"].append(assignment_values[i])

    df = pd.DataFrame(data)
    sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
    plt.show()