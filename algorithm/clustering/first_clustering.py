import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import random

k = 9
def cluster(init_data, now_unit): # 클러스터링
    data = {"x": [], "y": [], "cluster": [], "student": []}
    init_data = [float(i) for i in init_data]
    temp = 0
    vectors_set = init_data
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
    for step in range(100):
        count+=1
        a = sess.run(temp)
        b = sess.run(update_centroides)
        gap = 0 # 이전 클러스터링과의 차이값
        for i in range(len(a)):
            gap += (a[i]-b[i])**2
        gap /= len(a)
        if(gap<0.00000000000005): # 어느정도 정확도로 수렴한다면 멈춰준다
            break
    assignment_values = sess.run(assignments)
    for i in range(len(assignment_values)):
        data["x"].append(vectors_set[i])
        data["student"].append(i)
        data["y"].append(now_unit)
        data["cluster"].append(assignment_values[i])
    df = pd.DataFrame(data)
    sns.lmplot("x","y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
    plt.show()

    student = data["student"]
    unit = data["y"]
    cluster  = data["cluster"]
    matrix = [[] for i in range(k)]
    for i in range(len(init_data)):
        matrix[cluster[i]].append(student[i])
    print(matrix)

def load(change): # 텍스트 파일에서 이전 데이터를 불러온다
    data = []
    rf = open("test.txt",'r')
    linenum = 0
    while True:
        linenum+=1
        line = rf.readline()
        if not line:
            break
        if linenum != change[1]: # 해당하는 unit이 아닌  경우
            data += [line] # 원래 데이터를 그대로 넣어준다
        else: # 새로 들어온 데이터를 추가하여 넣어준다
            # check_s_id(change[0],len(line))
            newline = line[:-1]
            arr = newline.split(' ')
            arr += [str(change[2])]
            cluster(arr,change[1]) # 새로운 데이터로 클러스터링 진행
            newdata = ''
            for i in arr[:-1]:
                newdata = newdata + str(i) + ' '
            newdata += arr[len(arr)-1]
            newdata += '\n'
            data += [newdata]
    rf.close()
    wf = open("test.txt",'w')
    for i in data:
        wf.write(i)
    wf.close()

def start():
    for i in range(10):
        change = [10,5,random.random()] #s_id, unit, achivement
        load(change)
start()