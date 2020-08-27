from evaluators import gnet_evaluator

path_dict = {
    'annotations': '/ssd_scratch/cvit/kanishk/referit/cache/annotations/val.pickle'}
ckpt_path = '/ssd_scratch/cvit/kanishk/models/groundnet_vgg_bilstm_1x1_vg_final'
gnet_config = './configs/vgg_bilstm_1x1.yml'
geval = gnet_evaluator(gpu='0', ckpt_path=ckpt_path, data_path=path_dict,
                       gnet_config=gnet_config, query_level='referral')

[iou_acc,
 hit_acc,
 att_crr,
 wrd_idx_list,
 sen_idx_list,
 cat_lvl_scores,
 cat_cnt_correct,
 cat_cnt_correct_hit,
 cat_att_correct] = geval()
