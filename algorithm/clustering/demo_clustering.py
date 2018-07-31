import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import random
np.random.seed(1)
init_data = np.random.rand(2,500)
data = {"x": [], "y": [], "cluster": [], "student": []}
k = 9
for now in range(len(init_data)):
    temp = 0
    vectors_set = init_data[now]
    vectors = tf.constant(vectors_set)
    centroides = tf.Variable(tf.slice(tf.nn.top_k(tf.random_shuffle(vectors),len(vectors_set),True)[0],[0],[k]))
    expanded_vectors = tf.expand_dims(vectors, 0)
    expanded_centroides = tf.expand_dims(centroides, 1)

    # 할당 단계 : 유클리드 제곱거리 사용
    diff = tf.sub(expanded_vectors, expanded_centroides)
    sqr = tf.square(diff) # 제곱
    assignments = tf.argmin(sqr,0) # 최소값 인덱스
    # print(sess.run(assignments))
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
    past = tf.reduce_sum(tf.square(tf.sub(centroides,temp)))
    temp = centroides

    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)
    count = 0
    for step in range(50):
        count+=1
        a = sess.run(temp)
        b = sess.run(update_centroides)
        gap = 0
        for i in range(len(a)):
            gap += (a[i]-b[i])**2
        gap /= len(a)
        if(gap<0.00000000000005):
            break
    assignment_values = sess.run(assignments)
    print(count)
    for i in range(len(assignment_values)):
        data["x"].append(vectors_set[i])
        data["student"].append(i)
        data["y"].append(now)
        data["cluster"].append(assignment_values[i])
    
df = pd.DataFrame(data)
sns.lmplot("x","y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()

student = data["student"]
unit = data["y"]
cluster  = data["cluster"]
matrix = [[[] for j in range(k)] for i in range(len(init_data))]
for i in range(len(init_data)*len(init_data[0])):
    matrix[unit[i]][cluster[i]].append(student[i])
print(matrix)

# tf.concat(concat_dim,values)
# :텐서들을 하나의 차원에서 이어붙인다.
# t1 = [[1, 2, 3], [4, 5, 6]]
# t2 = [[7, 8, 9], [10, 11, 12]]
# tf.concat(0, [t1, t2]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
# tf.concat(1, [t1, t2]) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
