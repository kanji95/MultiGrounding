import tensorflow as tf
import os
import sys
import pickle

import numpy as np
from PIL import Image

import tensorflow.contrib.slim as slim
import tensorflow_hub as hub
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options,log_device_placement=True,allow_soft_placement=True)

ckpt_path = '/ssd_scratch/cvit/kanishk/models/groundnet_vgg_bilstm_1x1_vg_final'

annot_path = '/ssd_scratch/cvit/kanishk/referit/cache/annotations/val.pickle'
with open(annot_path, 'rb') as f:
    dict_val = pickle.load(f, encoding='latin1')

gamma_1 = 5.0
gamma_2 = 10.0
reg_val = .0005
num_tst = 500

def validate_referit(dict_test):
    cnt_overall = 0
    cnt_correct_w = 0
    cnt_correct_hit_w = 0
    cnt_correct_s = 0
    cnt_correct_hit_s = 0
    for k,doc_id in enumerate(dict_test):
        if k>num_tst:
            continue
        # imgbin = txn.get(doc_id.encode('utf-8'))
        # if imgbin==None:
        #     print ("Image not found")
        #     continue
        # buff = np.frombuffer(imgbin, dtype='uint8')
        # if len(buff) == 0:
        #     print ("Image not found")
        #     continue
        # imgbgr = cv2.imdecode(buff, cv2.IMREAD_COLOR)
        # imgrgb = imgbgr[:,:,[2,1,0]]

        # img = np.reshape(cv2.resize(imgrgb,(299,299)),(1,299,299,3))
        # orig_img_shape = dict_test[doc_id]['size'][:2]

        img_path = dict_test[doc_id]['img_path']
        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert(mode='RGB')
        img = np.array(img.resize((299, 299)))

        for i,annot in enumerate(dict_test[doc_id]['annotations']):
            if len(annot['bbox_norm'])== 0:
                continue
            if not check_percent(union(annot['bbox_norm'])):
                continue
            if any(b>1 for b in annot['bbox_norm']):
                continue
            unq_qry = set(annot['query'])
            sen_batch = [sen for sen in unq_qry if 0<len(sen.split())<=50] #only unique queries with 0<length<=50
            img_batch = np.repeat(img,len(sen_batch),axis=0)
            tensor_list = [heatmap_w, heatmap_s, R_i, R_s]
            feed_dict = {input_img: img_batch, text_batch: sen_batch, mode: 'test'}
            qry_heats, qry_heat, qry_scores, sen_score = sess.run(tensor_list, feed_dict)
            # add length of unique queries
            cnt_overall += len(sen_batch)
            for c,sen in enumerate(sen_batch):
                idx = [j for j in range(len(sen.split()))]
                if np.mean(qry_scores[c,idx])==0:
                    pred = {}
                else:
                    heatmap_wrd = np.average(qry_heats[c,idx,:], weights = qry_scores[c,idx], axis=0)
                    heatmap_sen = qry_heat[c,:]
                    bbox_c_w,hit_c_w = calc_correctness(annot,heatmap_wrd,orig_img_shape)
                    bbox_c_s,hit_c_s = calc_correctness(annot,heatmap_sen,orig_img_shape)
                    cnt_correct_w+=bbox_c_w
                    cnt_correct_hit_w+=hit_c_w
                    cnt_correct_s+=bbox_c_s
                    cnt_correct_hit_s+=hit_c_s

        var = [k,num_tst,cnt_correct_w/cnt_overall,cnt_correct_hit_w/cnt_overall]
        var_s = [cnt_correct_s/cnt_overall,cnt_correct_hit_s/cnt_overall]
        prnt0 = 'Sample {}/{}, IoU_acc_w:{:.2f}, IoU_acc_s:{:.2f}'.format(var[0],var[1],var[2],var_s[0])
        prnt1 = ', Hit_acc_w:{:.2f}, Hit_acc_s:{:.2f} \r'.format(var[3],var_s[1])
        sys.stdout.write(prnt0+prnt1)                
        sys.stdout.flush()

    hit_acc_w = cnt_correct_hit_w/cnt_overall
    iou_acc_w = cnt_correct_w/cnt_overall
    hit_acc_s = cnt_correct_hit_s/cnt_overall
    iou_acc_s = cnt_correct_s/cnt_overall
    
    return iou_acc_w,hit_acc_w,iou_acc_s,hit_acc_s

# def batch_gen(ids, annot_dict, txn):
#     img_batch = np.empty((n_batch, 299, 299, 3), dtype='float32')
#     cap_batch = []
#     #currently, it takes negative samples randomly from "all" dataset
#     #it randomly picks any batch, so it doesn't have ending
#     seen = {}
#     for i in range(n_batch):
#         choice_id = random.choice(ids)
#         while choice_id in seen: #we don't want to have repetitive img/caps in a batch 
#             choice_id = random.choice(ids)
#         imgbin = txn.get(choice_id.encode('utf-8'))
#         if imgbin!=None:
#             buff = np.frombuffer(imgbin, dtype='uint8')
#         else:
#             buff = []
#         while choice_id in seen or len(buff)==0:
#             choice_id = random.choice(ids)
#             imgbin = txn.get(choice_id.encode('utf-8'))
#             if imgbin!=None:
#                 buff = np.frombuffer(imgbin, dtype='uint8')
#             else:
#                 buff = []
#         seen[choice_id] = 1
        
#         imgbgr = cv2.imdecode(buff, cv2.IMREAD_COLOR)
#         img = imgbgr[:,:,[2,1,0]]
#         img_batch[i,:,:,:] = cv2.resize(img,(299,299))
#         queries = [annot['query'] for annot in dict_train[choice_id]['annotations']]
#         sentence = random.choice(queries)
#         cap_batch.append(sentence)
#     return img_batch, cap_batch

def attn_loss(e_w,v,e_s):
    #e: ?xTxD, v: ?xNx4xD, e_bar: ?xD
    with tf.variable_scope('attention_loss'):
        ###word-level###
        #heatmap
        h = tf.nn.relu(tf.einsum('bij,cklj->bcikl',e_w,v)) #pair-wise ev^T: ?x?xTxNx4
        #attention
        a = tf.einsum('bcijl,cjlk->bcikl',h,v) #?x?xTxDx4 attnded visual reps for each of T words for all pairs
        #pair-wise score
        a_norm = tf.nn.l2_normalize(a,axis=3)
        e_w_norm = tf.nn.l2_normalize(e_w,axis=2)
        R_ik = tf.einsum('bcilk,bil->bcik',a_norm,e_w_norm) #cosine for T (words,img_reps) for all pairs
        #level dropout
        #R_ik_sh = R_ik.get_shape().as_list()
        #R_ik = tf.layers.dropout(R_ik,rate=0.5,noise_shape=[1,1,1,R_ik_sh[3]],
        #                         training=isTraining)
        R_i = tf.reduce_max(R_ik,axis=-1) #?x?xT
        R = tf.log(tf.pow(tf.reduce_sum(tf.exp(gamma_1*R_i),axis=2),1/gamma_1)) #?x? cap-img pairs
        #posterior probabilities
        P_DQ = tf.diag_part(tf.nn.softmax(gamma_2*R,axis=0)) #P(cap match img)
        P_QD = tf.diag_part(tf.nn.softmax(gamma_2*R,axis=1)) #p(img match cap)
        #losses
        L1_w = -tf.reduce_mean(tf.log(P_DQ))
        L2_w = -tf.reduce_mean(tf.log(P_QD))
        
        ###sentence-level###
        #heatmap
        h_s = tf.nn.relu(tf.einsum('bj,cklj->bckl',e_s,v)) #pair-wise e_bar*v^T: ?x?xNx4
        #attention
        a_s = tf.einsum('bcjk,cjkl->bclk',h_s,v) #?x?xDx4 attnded visual reps for sen. for all pairs
        #pair-wise score
        a_s_norm = tf.nn.l2_normalize(a_s,axis=2)
        e_s_norm = tf.nn.l2_normalize(e_s,axis=1)
        R_sk = tf.einsum('bclk,bl->bck',a_s_norm,e_s_norm) #cosine for (sen,img_reps) for all pairs
        R_s = tf.reduce_max(R_sk,axis=-1) #?x?
        #posterior probabilities
        P_DQ_s = tf.diag_part(tf.nn.softmax(gamma_2*R_s,axis=0)) #P(cap match img)
        P_QD_s = tf.diag_part(tf.nn.softmax(gamma_2*R_s,axis=1)) #P(img match cap)
        #losses
        L1_s = -tf.reduce_mean(tf.log(P_DQ_s))
        L2_s = -tf.reduce_mean(tf.log(P_QD_s))
        #overall loss
        loss = L1_w + L2_w + L1_s + L2_s
    
    return loss
    
def attn(e_w,v,e_s):
    ## Inputs: local and global cap and img features ##
    ## Output: Heatmap for each word, Global Heatmap, Attnded Vis features, Corr-vals
    #e: ?xTxD, v: ?xNx4xD, e_bar: ?xD
    with tf.variable_scope('attention'):
        ###word-level###
        #heatmap pool
        h = tf.nn.relu(tf.einsum('bij,bklj->bikl',e_w,v)) #pair-wise ev^T: ?xTxNx4
        #attention
        a = tf.einsum('bijk,bjkl->bilk',h,v) #?xTxDx4 attnded visual reps for each of T words
        #pair-wise score
        a_norm = tf.nn.l2_normalize(a,axis=2)
        e_w_norm = tf.nn.l2_normalize(e_w,axis=2)
        R_ik = tf.einsum('bilk,bil->bik',a_norm,e_w_norm) #cosine for T (words,img_reps) for all pairs
        R_ik = tf.identity(R_ik,name='level_score_word')
        R_i = tf.reduce_max(R_ik,axis=-1,name='score_word') #?xT
        #R = tf.log(tf.pow(tf.reduce_sum(tf.exp(gamma_1*R_i),axis=1),1/gamma_1)) #? corrs
        #heatmap
        idx_i = tf.argmax(R_ik,axis=-1,name='level_index_word') #?xT index of the featuremap which maximizes R_i
        # with tf.name_scope('summaries'):
        #     tf.summary.histogram('histogram_w', idx_i)
        ii,jj = tf.meshgrid(tf.range(tf.shape(idx_i)[0]),tf.range(tf.shape(idx_i)[1]),indexing='ij')
        ii = tf.cast(ii,tf.int64)
        jj = tf.cast(jj,tf.int64)
        batch_idx_i = tf.stack([tf.reshape(ii,(-1,)),
                                tf.reshape(jj,(-1,)),
                                tf.reshape(idx_i,(-1,))],axis=1) #?Tx3 indices of argmax
        N0=int(np.sqrt(h.get_shape().as_list()[2]))
        h_max = tf.gather_nd(tf.transpose(h,[0,1,3,2]),batch_idx_i) #?TxN retrieving max heatmaps
        heatmap_wd = tf.reshape(h_max,[tf.shape(h)[0],tf.shape(h)[1],N0,N0],name='heatmap_word')
        heatmap_wd_l = tf.reshape(h,[tf.shape(h)[0],tf.shape(h)[1],N0,N0,tf.shape(h)[3]],name='level_heatmap_word')
        
        ###sentence-level###
        #heatmap pool
        h_s = tf.nn.relu(tf.einsum('bj,blkj->blk',e_s,v)) #pair-wise e_bar*v^T: ?xNx4
        #attention
        a_s = tf.einsum('bjk,bjki->bik',h_s,v) #?xDx4 attnded visual reps for sen.
        #pair-wise score
        a_s_norm = tf.nn.l2_normalize(a_s,axis=1)
        e_s_norm = tf.nn.l2_normalize(e_s,axis=1)
        R_sk = tf.einsum('bik,bi->bk',a_s_norm,e_s_norm) #cosine for (sen,img_reps)
        R_sk = tf.identity(R_sk,name='level_score_sentence')
        R_s = tf.reduce_mean(R_sk,axis=-1,name='score_sentence') #?
        #heatmap
        idx_k = tf.argmax(R_sk,axis=-1,name='level_index_sentence') #? index of the featuremap which maximizes R_i
        # with tf.name_scope('summaries'):
        #     tf.summary.histogram('histogram_s', idx_k)
        ii_k = tf.cast(tf.range(tf.shape(idx_k)[0]),dtype='int64')
        batch_idx_k = tf.stack([ii_k,idx_k],axis=1)
        N0_g=int(np.sqrt(h_s.get_shape().as_list()[1]))
        h_s_max = tf.gather_nd(tf.transpose(h_s,[0,2,1]),batch_idx_k) #?xN retrieving max heatmaps
        heatmap_sd = tf.reshape(h_s_max,[-1,N0_g,N0_g],name='heatmap_sentence')
        heatmap_sd_l = tf.reshape(h_s,[-1,N0_g,N0_g,tf.shape(h)[3]],name='level_heatmap_sentence')
        
    return heatmap_wd, heatmap_sd, R_i, R_s  

def add_1by1_conv(feat_map,n_layers,n_filters,name,regularizer):
    with tf.variable_scope(name+'_postConv'):
        for i in range(n_layers):
            with tf.variable_scope(name+'_stage_'+str(i)):
                feat_map = tf.layers.conv2d(feat_map,filters=n_filters[i],kernel_size=[1,1],kernel_regularizer=regularizer)
                feat_map = tf.nn.leaky_relu(feat_map,alpha=.25)
    return feat_map

def depth_selection(model):
    with tf.variable_scope('stack_v'):
        v1 = tf.identity(model['vgg_16/conv5/conv5_1'],name='v1')
        v1 = add_1by1_conv(v1,n_layers=3,n_filters=[1024,1024,1024],name='v1',regularizer=regularizer)
        size = v1.get_shape().as_list()[1:3]
        resize_method = tf.image.ResizeMethod.BILINEAR
        v2 = tf.identity(model['vgg_16/conv5/conv5_3'],name='v2')
        #v2 = tf.image.resize_images(v2, size, method=resize_method)
        v2 = add_1by1_conv(v2,n_layers=3,n_filters=[1024,1024,1024],name='v2',regularizer=regularizer)
        v3 = tf.identity(model['vgg_16/conv4/conv4_1'],name='v3')
        v3 = tf.image.resize_images(v3, size, method=resize_method, align_corners=True)
        v3 = add_1by1_conv(v3,n_layers=3,n_filters=[1024,1024,1024],name='v3',regularizer=regularizer)
        v4 = tf.identity(model['vgg_16/conv4/conv4_3'],name='v4')
        v4 = tf.image.resize_images(v4, size, method=resize_method, align_corners=True)
        v4 = add_1by1_conv(v4,n_layers=3,n_filters=[1024,1024,1024],name='v4',regularizer=regularizer)
        v_all = tf.stack([v1,v2,v3,v4], axis=3)
        v_all = tf.reshape(v_all,[-1,v_all.shape[1]*v_all.shape[2],v_all.shape[3],v_all.shape[4]])
        v_all = tf.nn.l2_normalize(v_all, axis=-1, name='stacked_image_feature_maps')
    return v_all

def build_bilstm(w_embd,seq_length):
    with tf.variable_scope('BiLSTM'):
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.LSTMCell(512, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.LSTMCell(512, forget_bias=1.0)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, w_embd, sequence_length = seq_length,
                                                  dtype=tf.float32)
    output = tf.concat(outputs,axis=2,name='BiLSTM_out')
    return output

sess = tf.InteractiveSession(config=config)

mode = tf.placeholder(tf.string, name='mode')
isTraining = tf.equal(mode, 'train')
regularizer = tf.contrib.layers.l2_regularizer(reg_val)

with tf.device('/gpu:1'):
    #building visual model
    print('Building Visual Model...')
    input_img = tf.placeholder(tf.float32, (None,299,299,3), name='input_img')
    pre_processed_img = pre_process(input_img, 'vgg_preprocessing')
    vis_model = pre_trained_load(model_name='vgg_16', image_shape=(None,299,299,3),
                              input_tensor=pre_processed_img, session=sess, is_training=False, global_pool=True)

    v = depth_selection(vis_model) #(?,1225,4,1024)
    
   #building text model
    print('Building Text Model...')
    #sentence placeholder - list of sentences
    text_batch = tf.placeholder('string', shape=[None], name='text_input')
    #loading pre-trained ELMo
    elmo = hub.Module("../modules/ELMo", trainable=True)
    #getting ELMo embeddings
    elmo_embds = elmo(text_batch, signature="default", as_dict=True)
    w_embd = tf.identity(elmo_embds['word_emb'], name='elmo_word_embd') #?xTxD/2
    lstm_embd = build_bilstm(w_embd,elmo_embds['sequence_len']) #?xTxD
    #taking index of last word in each sentence
    idx = elmo_embds['sequence_len']-1
    batch_idx = tf.stack([tf.range(0,tf.size(idx),1),idx],axis=1)
    # Concatenate first of backward with last of forward to get sentence embeddings
    dim = lstm_embd.get_shape().as_list()[-1]
    sen_embd = tf.concat([lstm_embd[:,0,int(dim/2):],
                            tf.gather_nd(lstm_embd[:,:,:int(dim/2)],batch_idx)], axis=-1) #[batch,dim]
    e_s = tf.layers.dense(sen_embd, units=1024)
    e_s = tf.nn.leaky_relu(e_s,alpha=.25)
    e_s = tf.layers.dense(e_s, units=1024)
    e_s = tf.nn.leaky_relu(e_s,alpha=.25)
    e_s = tf.nn.l2_normalize(e_s, axis=-1, name='sen_embedding')
    
    w_embd_tiled = tf.tile(w_embd,[1,1,2])
    w_embd = tf.concat([tf.expand_dims(w_embd_tiled,axis=3),tf.expand_dims(lstm_embd,axis=3)],axis=3)
    w_embd = tf.layers.dense(w_embd, units=1)[:,:,:,0]
    e_w = tf.layers.dense(w_embd, units=1024)
    e_w = tf.nn.leaky_relu(e_w,alpha=.25)
    e_w = tf.layers.dense(e_w, units=1024)
    e_w = tf.nn.leaky_relu(e_w,alpha=.25)
    e_w = tf.nn.l2_normalize(e_w, axis=-1, name='w_embedding')
    
    heatmap_w,heatmap_s,R_i,R_s = attn(e_w,v,e_s)

condition = 'BiLSTM_VGG_VG'
print('Initializing...')
_ = sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
#loading pretrained vgg weights
# print('Loading visual path model (vgg)...')
# vis_model.load_weights()
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
saver.restore(sess, ckpt_path)


iou_acc_w,hit_acc_w,iou_acc_s,hit_acc_s = validate_referit(dict_val)
sv = 'Word IOU:{}, Word Pointing Acc:{}, Sentence IOU:{}, Sentence Pointing Acc:{}\r'.format(iou_acc_w,hit_acc_w,iou_acc_s,hit_acc_s)
print(sv, flush=True)
