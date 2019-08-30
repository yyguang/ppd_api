#encoding:utf-8
import sys
import shutil

d = {
"毛泽东":"mao_ze_dong","江泽民":"jiang_ze_min","邓小平":"deng_xiao_ping","习近平":"xi_jin_ping","彭丽媛":"peng_li_yuan",
"李克强":"li_ke_qiang","金正恩":"jin_zheng_en","栗战书":"li_zhan_shu","汪洋":"wang_yang","王沪宁":"wang_hu_ning",
"赵乐际":"zhao_le_ji","韩正":"han_zheng","张德江":"zhang_de_jiang","俞正声":"yu_zheng_sheng","刘云山":"liu_yun_shan",
"王岐山":"wang_qi_shan","张高丽":"zhang_gao_li","胡锦涛":"hu_jin_tao","吴邦国":"wu_bang_guo","温家宝":"wen_jia_bao","贾庆林":"jia_qing_lin",
"李长春":"li_chang_chun","贺国强":"he_guo_qiang","周永康":"zhou_yong_kang","曾庆红":"zeng_qing_hong","黄菊":"huang_ju","吴官正":"wu_guan_zheng",
"罗干":"luo_gan","薄熙来":"bo_xi_lai","刘志军":"liu_zhi_jun","赵紫阳":"zhao_zi_yang","薄谷开来":"bo_gu_kai_lai","孟建柱":"meng_jian_zhu",
"徐才厚":"xu_cai_hou","朱镕基":"zhu_rong_ji","李洪志":"li_hong_zhi","王立军":"wang_li_jun","李鹏":"li_peng","鲁炜":"lu_wei",
"孙政才":"sun_zheng_cai", "胡春华": "hu_chun_hua","周恩来":"zhou_en_lai","奥巴马":"ao_ba_ma",
"特朗普": "te_lang_pu","蒋介石":"jiang_jie_shi","蔡英文": "cai_ying_wen","普京": "pu_jing", "本·拉登":"ben_la_deng","希特勒":"xi_te_le","巴格达迪":"ba_ge_da_di","柴玲":"chai_ling","陈水扁": "chen_shui_bian","达赖": "da_lai","封从德":"feng_cong_de","刘刚":"liu_gang","热比娅": "re_bi_ya","王丹":"wang_dan","吾尔开西":"wu_er_kai_xi", "戴晴":"dai_qing","狄玉明": "di_yu_ming","丁子霖": "ding_zi_lin","郭文贵": "guo_wen_gui","胡耀邦":"hu_yao_bang","李常受":"li_chang_shou","令计划": "ling_ji_hua","刘晓波": "liu_xiao_bo", "卢军宏":"lu_jun_hong","王维林":"wang_wei_lin","徐圣光": "xu_sheng_guang"}

input_dir = sys.argv[1]
#print input_dir
input_name = input_dir.split('/')[-1]
target_dir = d[input_name]
#print input_dir,input_dir.replace(input_name,target_dir)
shutil.move(input_dir,input_dir.replace(input_name,target_dir))
