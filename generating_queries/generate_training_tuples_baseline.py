import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

# 현 python 파일의 절대 경로를 얻는 코드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = '/home/ma/git/learning/PointNetVlad-Pytorch/dataset/oxford/'

runs_folder = ""
filename = "pointcloud_locations_20m_10overlap.csv"
pointcloud_fols = "/pointcloud_20m_10overlap/"

all_folders = sorted(os.listdir(os.path.join(BASE_DIR,base_path,runs_folder)))

folders = []

# All runs are used for training (both full and partial)

# Todo
# 0부터 전체 폴더 갯수-2 까지 <- 왜 한개를 더 빼지??
index_list = range(len(all_folders)-1)

# Folder list 생성
print("Number of runs: "+str(len(index_list)))
for index in index_list:
    folders.append(all_folders[index])
print(folders)


#####For training and test data split#####
x_width = 150
y_width = 150
p1 = [5735712.768124,620084.402381]
p2 = [5735611.299219,620540.270327]
p3 = [5735237.358209,620543.094379]
p4 = [5734749.303802,619932.693364]
p = [p1,p2,p3,p4]


# 
def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    
    # points내 4개의 point에 대해 
    for point in points:
        
        # 해당 point를 중심으로하는 가로 150 세로 150 크기의 정사각형 내에 들어올경우 in_test_set을 True로 설정
        if(point[0]-x_width < northing and northing < point[0]+x_width and point[1]-y_width < easting and easting < point[1]+y_width):
            in_test_set = True
            break
    return in_test_set
##########################################


def construct_query_dict(df_centroids, filename):
    """
    KDTree
    1. 하나의 축을 선정하여 해당 축을 따라 data point를 나눠 두 조각을 얻는다.
    2. 두 조각 각각에 대해 단일 point를 갖는 leaf node에 닿을때 까지 1번을 반복한다.
    3. 결과적으로 얻는 Tree가 Binary tree가 되고, 이때 각 node는 특정 axis를 따라 공간을 두개로 분할하는 hyperplane이 된다.
    """
    # KDTree 선언
    tree = KDTree(df_centroids[['northing','easting']])
    
    # 
    ind_nn = tree.query_radius(df_centroids[['northing','easting']],r=10)
    ind_r = tree.query_radius(df_centroids[['northing','easting']], r=50)
    queries = {}
    for i in range(len(ind_nn)):
        
        # iloc : 행 번호(row number)로 자료를 선택하는 방법
        query = df_centroids.iloc[i]["file"]
        
        # np.setdiff1d : 첫 배열 x로부터 두번째 배열 y를 뺀 차집합 반환
        # i번째 point의 positive 중 i번째는 제외
        positives = np.setdiff1d(ind_nn[i],[i]).tolist()
        
        # df_centorid의 index중, 50이내에도 들지 못한 index들
        negatives = np.setdiff1d(
            df_centroids.index.values.tolist(),ind_r[i]).tolist()
        
        # index 섞기 => permutation?
        random.shuffle(negatives)
        
        # triplet 만들기
        queries[i] = {"query":query,
                      "positives":positives,"negatives":negatives}

    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


# Initialize pandas DataFrame
# file, northing, easting을 column으로 갖는 두 DataFrame 생성
df_train = pd.DataFrame(columns=['file','northing','easting'])
df_test = pd.DataFrame(columns=['file','northing','easting'])

# 리스트로 만들어놓은 folders의 각 folder에 대해 연산 진행
for folder in folders:
    
    # dataframe path를 얻어 csv read
    df_locations = pd.read_csv(os.path.join(
        base_path,runs_folder,folder,filename),sep=',')
    
    # 'timestamp'에 해당하는 column내 정보를 얻어 path 형태로 전환
    df_locations['timestamp'] = runs_folder+folder + \
        pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
    
    # 'timestamp' column을 'file' column으로 대체
    df_locations = df_locations.rename(columns={'timestamp':'file'})

    for index, row in df_locations.iterrows():
        
        # p를 기준으로 testset, trainset 구분
        if(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
            
            # ignore_index : 기존 index를 무시하고 재설정 (0,1,2,3,...)
            df_test = df_test.append(row, ignore_index=True)
            
        else:
            df_train = df_train.append(row, ignore_index=True)

print("Number of training submaps: "+str(len(df_train['file'])))
print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))
construct_query_dict(df_train,"training_queries_baseline.pickle")
construct_query_dict(df_test,"test_queries_baseline.pickle")
