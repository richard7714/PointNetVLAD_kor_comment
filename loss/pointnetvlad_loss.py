import numpy as np
import math
import torch
import sys

def best_pos_distance(query, pos_vecs):
    
    # (2, 2, 256)
    num_pos = pos_vecs.shape[1]
    
    # query : (2, 1, 256) => 1번 차원에 대해 pos_vec 개수만큼 repeat 적용
    query_copies = query.repeat(1, int(num_pos), 1)
    
    # 두 vec간의 L2 Norm. sqrt가 되지않은
    diff = ((pos_vecs - query_copies) ** 2).sum(2)
    
    # diff 중 가장 작은 vect와 가장 큰 vect의 value 획득
    # Value, indices
    min_pos, _ = diff.min(1)
    max_pos, _ = diff.max(1)
    
    return min_pos, max_pos


def triplet_loss(q_vec, pos_vecs, neg_vecs, margin, use_min=False, lazy=False, ignore_zero_loss=False):
    
    # pos_vecs 내에서 q_vec과 가장 가까운 vect하나와 가장 먼 vect하나를 획득
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    # neg vect의 개수
    num_neg = neg_vecs.shape[1]
    
    # batch의 크기
    batch = q_vec.shape[0]
    
    # q_vec를 num_neg만큼 repeat한 matrix 생성
    query_copies = q_vec.repeat(1, int(num_neg), 1)
    
    # positive를 2차원으로 만들고, num_neg만큼 반복하여 2차원 행렬 생성
    # (2) => (2, 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))
    
    loss = margin + positive - ((neg_vecs - query_copies) ** 2).sum(2)
    
    # clamp : 주어진 범위에서 값이 벗어나면 그 값을 해당 범위의 최소값 또는 최대값으로 잘라내는 역할 수행
    loss = loss.clamp(min=0.0)
    
    if lazy:
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.sum(1)
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()
    return triplet_loss


def triplet_loss_wrapper(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):
    return triplet_loss(q_vec, pos_vecs, neg_vecs, m1, use_min, lazy, ignore_zero_loss)


def quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):
    
    # pos_vecs 내에서 q_vec과 가장 가까운 vect하나와 가장 먼 vect하나를 획득
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    # neg vect의 개수
    num_neg = neg_vecs.shape[1]
    
    # batch의 크기    
    batch = q_vec.shape[0]
    
    # q_vec를 num_neg만큼 repeat한 matrix 생성
    query_copies = q_vec.repeat(1, int(num_neg), 1)
    
    # positive를 2차원으로 만들고, num_neg만큼 반복하여 2차원 행렬 생성
    # (2) => (2, 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))

    loss = m1 + positive - ((neg_vecs - query_copies) ** 2).sum(2)
    
    # clamp : 주어진 범위에서 값이 벗어나면 그 값을 해당 범위의 최소값 또는 최대값으로 잘라내는 역할 수행    
    loss = loss.clamp(min=0.0)
    
    # 기존의 distance에서 Tuple 내 모든 pointcloud와 유사하지 않은 negative Pointcloud와 Tuple 내 Negative PCL간의 거리를 최대화
    if lazy:
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.sum(1)
    
    # 
    if ignore_zero_loss:
        # 크면 True, 작으면 False. Hard Assignment인듯
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        
        # True의 갯수
        num_hard_triplets = torch.sum(hard_triplets)
        
        # 전체 Loss의 평균
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()

    # P_neg*
    other_neg_copies = other_neg.repeat(1, int(num_neg), 1)
    
    # second loss 계산
    second_loss = m2 + positive - ((neg_vecs - other_neg_copies) ** 2).sum(2)
    second_loss = second_loss.clamp(min=0.0)
    
    # Todo
    if lazy:
        second_loss = second_loss.max(1)[0]
    else:
        second_loss = second_loss.sum(1)

    # Hard assign or 그냥
    if ignore_zero_loss:
        hard_second = torch.gt(second_loss, 1e-16).float()
        num_hard_second = torch.sum(hard_second)
        second_loss = second_loss.sum() / (num_hard_second + 1e-16)
    else:
        second_loss = second_loss.mean()

    total_loss = triplet_loss + second_loss
    return total_loss
