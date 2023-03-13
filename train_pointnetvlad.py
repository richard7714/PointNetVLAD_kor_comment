import argparse
import importlib
import math
import os
import socket
import sys

import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors

import config as cfg
import evaluate
import loss.pointnetvlad_loss as PNV_loss
import models.PointNetVlad as PNV
import torch
import torch.nn as nn
from loading_pointclouds import *
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.backends import cudnn


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)



cudnn.enabled = True

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log/', help='Log dir [default: log]')
parser.add_argument('--results_dir', default='results/',
                    help='results dir [default: results]')
parser.add_argument('--positives_per_query', type=int, default=2,
                    help='Number of potential positives in each training tuple [default: 2]')
parser.add_argument('--negatives_per_query', type=int, default=18,
                    help='Number of definite negatives in each training tuple [default: 18]')
parser.add_argument('--max_epoch', type=int, default=20,
                    help='Epoch to run [default: 20]')
parser.add_argument('--batch_num_queries', type=int, default=2,
                    help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.000005,
                    help='Initial learning rate [default: 0.000005]')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000,
                    help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7,
                    help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--margin_1', type=float, default=0.5,
                    help='Margin for hinge loss [default: 0.5]')
parser.add_argument('--margin_2', type=float, default=0.2,
                    help='Margin for hinge loss [default: 0.2]')
parser.add_argument('--loss_function', default='quadruplet', choices=[
                    'triplet', 'quadruplet'], help='triplet or quadruplet [default: quadruplet]')
parser.add_argument('--loss_not_lazy', action='store_true',
                    help='If present, do not use lazy variant of loss')
parser.add_argument('--loss_ignore_zero_batch', action='store_true',
                    help='If present, mean only batches with loss > 0.0')
parser.add_argument('--triplet_use_best_positives', action='store_true',
                    help='If present, use best positives, otherwise use hardest positives')
parser.add_argument('--resume', action='store_true',
                    help='If present, restore checkpoint and resume training')
parser.add_argument('--dataset_folder', default='../../dataset/',
                    help='PointNetVlad Dataset Folder')

FLAGS = parser.parse_args()
cfg.BATCH_NUM_QUERIES = FLAGS.batch_num_queries
#cfg.EVAL_BATCH_SIZE = 12
cfg.NUM_POINTS = 4096
cfg.TRAIN_POSITIVES_PER_QUERY = FLAGS.positives_per_query
cfg.TRAIN_NEGATIVES_PER_QUERY = FLAGS.negatives_per_query
cfg.MAX_EPOCH = FLAGS.max_epoch
cfg.BASE_LEARNING_RATE = FLAGS.learning_rate
cfg.MOMENTUM = FLAGS.momentum
cfg.OPTIMIZER = FLAGS.optimizer
cfg.DECAY_STEP = FLAGS.decay_step
cfg.DECAY_RATE = FLAGS.decay_rate
cfg.MARGIN1 = FLAGS.margin_1
cfg.MARGIN2 = FLAGS.margin_2
cfg.FEATURE_OUTPUT_DIM = 256

cfg.LOSS_FUNCTION = FLAGS.loss_function
cfg.TRIPLET_USE_BEST_POSITIVES = FLAGS.triplet_use_best_positives
cfg.LOSS_LAZY = FLAGS.loss_not_lazy
cfg.LOSS_IGNORE_ZERO_BATCH = FLAGS.loss_ignore_zero_batch

cfg.TRAIN_FILE = 'generating_queries/training_queries_refine.pickle'
cfg.TEST_FILE = 'generating_queries/test_queries_refine.pickle'

cfg.LOG_DIR = FLAGS.log_dir
if not os.path.exists(cfg.LOG_DIR):
    os.mkdir(cfg.LOG_DIR)
LOG_FOUT = open(os.path.join(cfg.LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

cfg.RESULTS_FOLDER = FLAGS.results_dir

cfg.DATASET_FOLDER = FLAGS.dataset_folder

# Load dictionary of training queries
TRAINING_QUERIES = get_queries_dict(cfg.TRAIN_FILE)
TEST_QUERIES = get_queries_dict(cfg.TEST_FILE)

cfg.BN_INIT_DECAY = 0.5
cfg.BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(cfg.DECAY_STEP)
cfg.BN_DECAY_CLIP = 0.99

HARD_NEGATIVES = {}
TRAINING_LATENT_VECTORS = []

TOTAL_ITERATIONS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
# 0.5 * (0.5 ** (batch * 2 // 200000))
# Batch가 증가할때마다 momentum 감소
# 계속 감소하다가 BN_DECAY_CLIP 밑으로 떨어지면 CLIP이 return
def get_bn_decay(batch):
    bn_momentum = cfg.BN_INIT_DECAY * \
        (cfg.BN_DECAY_DECAY_RATE **
         (batch * cfg.BATCH_NUM_QUERIES // BN_DECAY_DECAY_STEP))
    return min(cfg.BN_DECAY_CLIP, 1 - bn_momentum)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

# learning rate halfed every 5 epoch


# Learning rate decay
# Epoch이 진행됨에 따라 Learning rate가 감소한다. 하지만 0.00001이하로는 떨어지지 않는다.
def get_learning_rate(epoch):
    learning_rate = cfg.BASE_LEARNING_RATE * ((0.9) ** (epoch // 5))
    learning_rate = max(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def train():
    global HARD_NEGATIVES, TOTAL_ITERATIONS
    bn_decay = get_bn_decay(0)
    #tf.summary.scalar('bn_decay', bn_decay)

    #loss = lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
    if cfg.LOSS_FUNCTION == 'quadruplet':
        loss_function = PNV_loss.quadruplet_loss
    else:
        loss_function = PNV_loss.triplet_loss_wrapper
    learning_rate = get_learning_rate(0)

    train_writer = SummaryWriter(os.path.join(cfg.LOG_DIR, 'train'))
    #test_writer = SummaryWriter(os.path.join(cfg.LOG_DIR, 'test'))

    # Global Feature 형태로 return, Residual 연산 수행, output_dim = 256, num_points = 4096
    model = PNV.PointNetVlad(global_feat=True, feature_transform=True,
                             max_pool=False, output_dim=cfg.FEATURE_OUTPUT_DIM, num_points=cfg.NUM_POINTS)
    model = model.to(device)

    # Model Parameter중 requires_grad가 true인 것들만 통과
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # Config에 따라 optimizer 설정
    if cfg.OPTIMIZER == 'momentum':
        optimizer = torch.optim.SGD(
            parameters, learning_rate, momentum=cfg.MOMENTUM)
    elif cfg.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(parameters, learning_rate)
    else:
        optimizer = None
        exit(0)

    # Checkpoint에서 재개할 때 사용
    if FLAGS.resume:
        resume_filename = cfg.LOG_DIR + "checkpoint.pth.tar"
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        
        # 기록된 parameter 획득
        saved_state_dict = checkpoint['state_dict']
        
        # 기록된 epoch 획득
        starting_epoch = checkpoint['epoch']
        
        # checkpoint 시점까지 이미 진행된 iteration
        TOTAL_ITERATIONS = starting_epoch * len(TRAINING_QUERIES)

        model.load_state_dict(saved_state_dict)
        
        # 'optimizer'에 해당하는 paramter 획득
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        starting_epoch = 0

    # data 병렬처리
    model = nn.DataParallel(model)

    LOG_FOUT.write(cfg.cfg_str())
    LOG_FOUT.write("\n")
    LOG_FOUT.flush()

    # Training 진행
    for epoch in range(starting_epoch, cfg.MAX_EPOCH):
        print(epoch)
        print()
        log_string('**** EPOCH %03d ****' % (epoch))
        
        # buffering에 의해 출력이 올바르게 나오지 않는 것을 방지
        sys.stdout.flush()  

        # train_one_epoch(model, optimizer, train_writer, loss_function, epoch)

        log_string('EVALUATING...')
        cfg.OUTPUT_FILE = cfg.RESULTS_FOLDER + 'results_' + str(epoch) + '.txt'
        eval_recall = evaluate.evaluate_model(model)
        log_string('EVAL RECALL: %s' % str(eval_recall))

        train_writer.add_scalar("Val Recall", eval_recall, epoch)


def train_one_epoch(model, optimizer, train_writer, loss_function, epoch):
    global HARD_NEGATIVES
    global TRAINING_LATENT_VECTORS, TOTAL_ITERATIONS

    is_training = True
    sampled_neg = 4000
    # number of hard negatives in the training tuple
    # which are taken from the sampled negatives
    num_to_take = 10

    # Shuffle train files
    # 0 ~ len()까지의 범위 내 step=1로 idx 생성
    train_file_idxs = np.arange(0, len(TRAINING_QUERIES.keys()))
    
    # shuffle
    np.random.shuffle(train_file_idxs)

    # batch의 갯수로 전체 idx를 나눈 후, training 진행
    # 현재 batch는 2
    for i in range(len(train_file_idxs)//cfg.BATCH_NUM_QUERIES):
        # for i in range (5):
        
        # i * 2 : (i+1)  * 2
        # batch 갯수만큼 training key를 불러온다
        batch_keys = train_file_idxs[i *
                                     cfg.BATCH_NUM_QUERIES:(i+1)*cfg.BATCH_NUM_QUERIES]
        q_tuples = []

        faulty_tuple = False    
        no_other_neg = False
        
        # batch 갯수만큼 iteration 진행
        for j in range(cfg.BATCH_NUM_QUERIES):  
            
            # batch_keys[j]에 해당하는 idx의 positive가 지정한 threshold보다 적다면 tuple이 적절치 못하다 판단하고 학습하지 않는다.
            if (len(TRAINING_QUERIES[batch_keys[j]]["positives"]) < cfg.TRAIN_POSITIVES_PER_QUERY):
                faulty_tuple = True
                break

            # no cached feature vectors
            # 가지고 있는 training latent vector가 없을 경우 == epoch을 처음 시작했을때
            if (len(TRAINING_LATENT_VECTORS) == 0):
                q_tuples.append(
                    # query, pos, neg, neg2
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY,
                                    TRAINING_QUERIES, hard_neg=[], other_neg=True))

            # TODO
            # 이런 경우가 언제 발생하는지? 
            elif (len(HARD_NEGATIVES.keys()) == 0):
                query = get_feature_representation(
                    TRAINING_QUERIES[batch_keys[j]]['query'], model)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]
                                             ]['negatives'][0:sampled_neg]
                hard_negs = get_random_hard_negatives(
                    query, negatives, num_to_take)
                print(hard_negs)
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY,
                                    TRAINING_QUERIES, hard_negs, other_neg=True))
            else:
                query = get_feature_representation(
                    TRAINING_QUERIES[batch_keys[j]]['query'], model)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]
                                             ]['negatives'][0:sampled_neg]
                hard_negs = get_random_hard_negatives(
                    query, negatives, num_to_take)
                
                # set() : 중복을 허용하지 않는다. 집합이므로
                # union() : 인자로 받은 두 list를 합친다.
                # 
                hard_negs = list(set().union(
                    HARD_NEGATIVES[batch_keys[j]], hard_negs))
                print('hard', hard_negs)
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY,
                                    TRAINING_QUERIES, hard_negs, other_neg=True))

            # j번째 batch_idx의 neg2에 해당하는 pointcloud 개수가 cfg와 다르다면 neg2가 없는 것을 판단
            if (q_tuples[j][3].shape[0] != cfg.NUM_POINTS):
                no_other_neg = True
                break
        
        # faulty tuple이 true이면 패스
        if(faulty_tuple):
            log_string('----' + str(i) + '-----')
            log_string('----' + 'FAULTY TUPLE' + '-----')
            continue
        
        # neg2가 없으면 패스
        if(no_other_neg):
            log_string('----' + str(i) + '-----')
            log_string('----' + 'NO OTHER NEG' + '-----')
            continue
        
        queries = []
        positives = []
        negatives = []
        other_neg = []
        
        # 얻은 tuple을 각각의 list에 입력
        for k in range(len(q_tuples)):
            queries.append(q_tuples[k][0])
            positives.append(q_tuples[k][1])
            negatives.append(q_tuples[k][2])
            other_neg.append(q_tuples[k][3])

        queries = np.array(queries, dtype=np.float32)
        # 1번 위치에 크기 1의 차원 추가
        queries = np.expand_dims(queries, axis=1)
        
        other_neg = np.array(other_neg, dtype=np.float32)
        other_neg = np.expand_dims(other_neg, axis=1)
        
        positives = np.array(positives, dtype=np.float32)
        negatives = np.array(negatives, dtype=np.float32)
        
        log_string('----' + str(i) + '-----')
        if (len(queries.shape) != 4):
            log_string('----' + 'FAULTY QUERY' + '-----')
            continue

        model.train()
        optimizer.zero_grad()

        output_queries, output_positives, output_negatives, output_other_neg = run_model(
            model, queries, positives, negatives, other_neg)
        loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, cfg.MARGIN_1, cfg.MARGIN_2, use_min=cfg.TRIPLET_USE_BEST_POSITIVES, lazy=cfg.LOSS_LAZY, ignore_zero_loss=cfg.LOSS_IGNORE_ZERO_BATCH)
        
        # backpropagation
        loss.backward()
        
        # backprop을 통해 얻은 변화도에 따라 파라미터 optimize
        optimizer.step()

        log_string('batch loss: %f' % loss)
        
        # loss를 cpu로 옮기고 item()을 통해 python의 숫자 형태로 변환
        train_writer.add_scalar("Loss", loss.cpu().item(), TOTAL_ITERATIONS)
        TOTAL_ITERATIONS += cfg.BATCH_NUM_QUERIES

        # EVALLLL

        # 조건의 의미? => github issue 참고
        if (epoch > 5 and i % (1400 // cfg.BATCH_NUM_QUERIES) == 29):
            TRAINING_LATENT_VECTORS = get_latent_vectors(
                model, TRAINING_QUERIES)
        print("Updated cached feature vectors")

        # 
        if (i % (6000 // cfg.BATCH_NUM_QUERIES) == 101):
            
            # model에 nn.DataParallel로 래핑되어있는 경우 실제 모델이 아닌 모듈 객체가 저장될 수 있는 경우를 방지하기 위함
            # nn.DataParallel로 래핑된 경우 래핑을 해제하고 실제 모델을 저장하기 위해 model.module을 이용함
            if isinstance(model, nn.DataParallel):
                model_to_save = model.module
            else:
                model_to_save = model
            save_name = cfg.LOG_DIR + cfg.MODEL_FILENAME
            torch.save({
                'epoch': epoch,
                'iter': TOTAL_ITERATIONS,
                'state_dict': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
                save_name)
            print("Model Saved As " + save_name)


def get_feature_representation(filename, model):
    model.eval()
    queries = load_pc_files([filename])
    
    # axis 번째에 차원추가
    queries = np.expand_dims(queries, axis=1)

    # query model 통과 후 representation 획득
    with torch.no_grad():
        q = torch.from_numpy(queries).float()
        q = q.to(device)
        output = model(q)
    output = output.detach().cpu().numpy()
    output = np.squeeze(output)
    model.train()
    return output

# random_negs : negative의 idx가 random하게 나열된 배열
def get_random_hard_negatives(query_vec, random_negs, num_to_take):
    
    # TRAINING_LATENT_VECTORS라는 전역변수를 사용
    global TRAINING_LATENT_VECTORS

    latent_vecs = []
    
    # random_negs에 담긴 idx를 선택하여 해당하는 latent vector를 latent_vecs에 append
    for j in range(len(random_negs)):
        latent_vecs.append(TRAINING_LATENT_VECTORS[random_negs[j]])

    latent_vecs = np.array(latent_vecs)
    
    # latent_vecs에 대한 KDTree 형성
    nbrs = KDTree(latent_vecs)
    
    # query_vec에 대해 num_to_take만큼의 indice를 추출
    distances, indices = nbrs.query(np.array([query_vec]), k=num_to_take)
    
    # TODO
    # 차원 체크 필요
    hard_negs = np.squeeze(np.array(random_negs)[indices[0]])
    
    hard_negs = hard_negs.tolist()
    return hard_negs


def get_latent_vectors(model, dict_to_process):
    
    # 0 ~ len(dict_to_process.keys())개의 1 step index list
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    # batch 내에 들어있는 pcd 파일의 개수
    batch_num = cfg.BATCH_NUM_QUERIES * \
        (1 + cfg.TRAIN_POSITIVES_PER_QUERY + cfg.TRAIN_NEGATIVES_PER_QUERY + 1)
    q_output = []

    model.eval()


    for q_index in range(len(train_file_idxs)//batch_num):
        
        # batch_num 만큼의 index
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        
        # 실제 파일 directory를 file_names에 담음
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        
        # file_names에 담긴 pcd를 queries에 입력
        queries = load_pc_files(file_names)

        feed_tensor = torch.from_numpy(queries).float()
        
        # [pcd개수, 4096, 3] => [pcd개수, 1, 4096, 3]
        feed_tensor = feed_tensor.unsqueeze(1)
        feed_tensor = feed_tensor.to(device)
        with torch.no_grad():
            out = model(feed_tensor)

        out = out.detach().cpu().numpy()
        
        # size가 1인 차원 삭제
        out = np.squeeze(out)

        q_output.append(out)

    q_output = np.array(q_output)
    
    # q_output으로 하나라도 얻었다면, 마지막이 feature vector 크기가 되도록 reshape
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # batch_num으로 나누었을 때 하나의 batch를 형성하지 못한 끝부분에 대한 연산
    # len(train_file_idxs) // batch_num * batch_num => idx를 batch_num의 배수로 만듬, 6671 => 6644 : 44의 배수
    # 6644 ~ 6671은 하나의 batch를 만들지 채우지 못하는 edge case
    # handle edge case
    # dict_to_process.keys() == 전체 point의 갯수
    # batch로 나누어지지 못한 것들은 따로따로 뽑아낸다.
    
    for q_index in range((len(train_file_idxs) // batch_num * batch_num), len(dict_to_process.keys())):
        index = train_file_idxs[q_index]
        queries = load_pc_files([dict_to_process[index]["query"]])
        queries = np.expand_dims(queries, axis=1)

        with torch.no_grad():
            queries_tensor = torch.from_numpy(queries).float()
            o1 = model(queries_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    return q_output


def run_model(model, queries, positives, negatives, other_neg, require_grad=True):
    
    # numpy to tensor
    queries_tensor = torch.from_numpy(queries).float()
    positives_tensor = torch.from_numpy(positives).float()
    negatives_tensor = torch.from_numpy(negatives).float()
    other_neg_tensor = torch.from_numpy(other_neg).float()
    
    # 1번 차원에 대해 concat (1+2+18+1 => 22), (2, 22, 4096, 3)
    feed_tensor = torch.cat(
        (queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor), 1)
    
    # (2, 22, 4096, 3) => (44, 1, 4096, 3)
    feed_tensor = feed_tensor.view((-1, 1, cfg.NUM_POINTS, 3))
    feed_tensor.requires_grad_(require_grad)
    feed_tensor = feed_tensor.to(device)
    if require_grad:
        output = model(feed_tensor)
    else:
        # 연산속도를 빠르게 하기 위함?
        with torch.no_grad():
            output = model(feed_tensor)
        
    # batch개수, feature dimension만 맞추고 나머지는 알아서 계산
    # (44, 256) -> (2, 22, 256)
    output = output.view(cfg.BATCH_NUM_QUERIES, -1, cfg.FEATURE_OUTPUT_DIM)
    
    # 1번 차원에 대해 output을 list에 나열된 숫자 개수에 따라 분할
    # (2, 22, 256) => (2, 1, 256), (2, 2, 256), (2, 18, 256), (2, 1, 256)
    o1, o2, o3, o4 = torch.split(
        output, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=1)

    return o1, o2, o3, o4


if __name__ == "__main__":
    train()