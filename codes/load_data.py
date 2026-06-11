import dgl
import torch as th
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from utils import calc_pairwise_cosine_similarity

def load_dataset(args , exclude_dr_di_edges=None):
    """تحميل الشبكة غير المتجانسة لبيانات الأدوية والأمراض"""

    # 1. تحميل مصفوفات التشابه الأساسية (الجراف الأول g)
    dr_dr = pd.read_csv(f'../data/{args.dataset}/drug_sim.csv', header=None).values
  
   # اضافة .copy() لضمان استقلالية البيانات  لانه من الخطا تعديل المصفوفه الاصليه 
    dr_sim = dr_dr.copy()

# هذا الجزء لتحويل Similarity Matrices الى Edges
    # 
    for i in range(len(dr_dr)):
        sorted_idx = np.argpartition(dr_sim[i], 15) # عزل اكبر 15 قيمة تشابه 
        dr_dr[i, sorted_idx[-15:]] = 1   # لانشاء الرابط تضع 1 عند ال15 دواء الاعلى تشابه لتحويل المصفوفه الى روابط (0,1)
    dr_dr = pd.DataFrame(np.array(np.where(dr_dr == 1)).T, columns=['Drug1', 'Drug2'])  #.T (Transpose): تقوم بقلب المصفوفة الناتجة لتصبح على شكل أزواج مرتبة [Drug1, Drug2].

#pd.DataFrame....
#np.where(dr_dr == 1): المصفوفة الأصلية ضخمة ومليئة بالأصفار. هذه الدالة تعمل كـ "كشّاف" يبحث فقط عن الرقم (1) ويستخرج إحداثياته (رقم الصف ورقم العمود).
#.T (Transpose): تقوم بقلب المصفوفة الناتجة لتصبح على شكل أزواج مرتبة. بدلاً من مصفوفة مربعة، يصبح لدينا جدول به عمودان فقط: "الدواء المصدر" و"الدواء الهدف".
#هذا التحويل يسمى Sparse Representation (التمثيل الخفيف). هو يقلل استهلاك الذاكرة بشكل هائل ويجعل الجراف جاهزاً للمعالجة بواسطة مكتبة DGL.

    di_di = pd.read_csv(f'../data/{args.dataset}/dis_sim.csv', header=None).values
    di_sim = di_di.copy() # هنا ايضا استخدمت copy 

    for i in range(len(di_di)):
        sorted_idx = np.argpartition(di_sim[i], 15)
        di_di[i, sorted_idx[-15:]] = 1
    di_di = pd.DataFrame(np.array(np.where(di_di == 1)).T, columns=['Disease1', 'Disease2'])

    dr_di_raw = pd.read_csv(f'../data/{args.dataset}/drug_dis.csv', header=None) #اضافة كلمة raw  , لتمييز المصفوفه الكامله عن قائمة الروابط لتجنب ال overwriting 
    dr_di = pd.DataFrame(np.array(np.where(dr_di_raw == 1)).T, columns=['Drug', 'Disease'])


    # --- الكود المصحح ---
    # نستخرج جميع الإحداثيات الموجبة أولاً
    all_pos_coords = np.array(np.where(dr_di_raw.values == 1)).T
    
    # إذا طُلب استبعاد روابط معينة (مثلاً روابط الاختبار أو الروابط المخفية)
    # نقوم بتصفيتها هنا قبل بناء الجراف
    if exclude_dr_di_edges is not None and len(exclude_dr_di_edges) > 0:
        exclude_set = set(map(tuple, exclude_dr_di_edges))
        mask = np.array([tuple(coord) not in exclude_set for coord in all_pos_coords])
        filtered_coords = all_pos_coords[mask]
    else:
        filtered_coords = all_pos_coords
    
    dr_di = pd.DataFrame(filtered_coords, columns=['Drug', 'Disease'])
    
 # بناء الجراف الأساسي g

    graph_data = {
        ('drug', 'dr_dr', 'drug'): (th.tensor(dr_dr['Drug1'].values), th.tensor(dr_dr['Drug2'].values)),
        ('disease', 'di_di', 'disease'): (th.tensor(di_di['Disease1'].values), th.tensor(di_di['Disease2'].values)),
        ('drug', 'dr_di', 'disease'): (th.tensor(dr_di['Drug'].values), th.tensor(dr_di['Disease'].values)),
        ('disease', 'di_dr', 'drug'): (th.tensor(dr_di['Disease'].values), th.tensor(dr_di['Drug'].values)), #روابط عكسيه للسطر السابق لاخبار الجراف انها في الاتجاهين 
    }

 #تحويل العلاقات الى جراف غير متجانس بنوعين منن العقد 
    g = dgl.heterograph(graph_data)

 # للأدوية: نضع مصفوفة تشابه الأدوية (dr_sim) في البداية، ثم نملأ الجزء المتبقي بالأصفار (بعدد الأمراض).
 #للأمراض: نضع أصفاراً في البداية (بعدد الأدوية)، ثم نضع مصفوفة تشابه الأمراض (di_sim).
 #الهدف: توحيد طول المتجهات (Vectors) لكل العقد.
    dr_feat_init = np.hstack((dr_sim, np.zeros((g.num_nodes('drug'), g.num_nodes('disease')))))
    di_feat_init = np.hstack((np.zeros((g.num_nodes('disease'), g.num_nodes('drug'))), di_sim))
   

   #تخزين هذه البيانات في المتغير h يسمح لنموذج الـ GNN في الخطوات التالية بالبدء بعملية الـ Message Passing، حيث تنتقل ميزات الأدوية وتتشابك مع ميزات الأمراض عبر الروابط التي بنيناها."
    g.nodes['drug'].data['h'] = th.from_numpy(dr_feat_init).to(th.float32)
    g.nodes['disease'].data['h'] = th.from_numpy(di_feat_init).to(th.float32)


    # بناء الجراف الثاني (g_llm) بناءً على الـ Embeddings المختارة
    # "حجز" مكان للمتغيرات
    #في الكود القديم: كان يعتمد على "الحقن المباشر" (Direct Injection) من الـ Terminal إلى الدوال، وإذا لم يجد الملف قد ينهار البرنامج فجأة (Crash).
    #اضافة هذا السطر هو لحجز مكان لها ثم التعبئه ستتم بناء على الif مما يجعل الكود  أكثر استقراراً (Robust).

    dr_feat_emb, di_feat_emb = None, None
    
    # --- المسار الجديد: ChemBERTa ---
    if args.feature_type == 'ChemBERTa':
        print(">>> Loading ChemBERTa chemical embeddings for drugs...")
        with open(f'../feat/{args.dataset}/ChemBERTa_drug_emb.pkl', 'rb') as f:
            dr_emb_dict = pickle.load(f)
            
        # ترتيب الأدوية لضمان تطابق المصفوفة مع نود الجراف
        #يضمن أن أول دواء في المصفوفة هو نفسه الدواء رقم 0 في الجراف، لتجنب خلط البيانات.
        dr_feat_emb = th.tensor([dr_emb_dict[i] for i in sorted(dr_emb_dict.keys())]).to(th.float32)
        
        # للأمراض نستخدم BERT لعدم وجود SMILES
        with open(f'../feat/{args.dataset}/BERT_disease_emb.pkl', 'rb') as f:
            di_emb_dict = pickle.load(f)
        di_feat_emb = th.tensor(list(di_emb_dict.values())).to(th.float32)

    # --- المسارات القديمة: تبقى كما هي ---
    elif args.BERT_emb:
        with open(f'../feat/{args.dataset}/BERT_drug_emb.pkl', 'rb') as f:
            dr_feat_emb = th.tensor(list(pickle.load(f).values())).to(th.float32)
        with open(f'../feat/{args.dataset}/BERT_disease_emb.pkl', 'rb') as f:
            di_feat_emb = th.tensor(list(pickle.load(f).values())).to(th.float32)
    
    elif args.LLM_emb:
        with open(f'../feat/{args.dataset}/LLM_drug_emb.pkl', 'rb') as f:
            dr_feat_emb = th.tensor(list(pickle.load(f).values())).to(th.float32)
        with open(f'../feat/{args.dataset}/LLM_disease_emb.pkl', 'rb') as f:
            di_feat_emb = th.tensor(list(pickle.load(f).values())).to(th.float32)

    # حساب مصفوفة التشابه الجديدة لـ g_llm إذا تم اختيار أحد الخيارات أعلاه
    if dr_feat_emb is not None:
        dr_sim_new = calc_pairwise_cosine_similarity(dr_feat_emb)
        di_sim_new = calc_pairwise_cosine_similarity(di_feat_emb)
        
        for i in range(len(dr_sim_new)):
            sorted_idx = np.argpartition(dr_sim_new[i], 15)
            dr_sim_new[i, sorted_idx[-15:]] = 1
        for i in range(len(di_sim_new)):
            sorted_idx = np.argpartition(di_sim_new[i], 15)
            di_sim_new[i, sorted_idx[-15:]] = 1
            
        dr_dr_new = pd.DataFrame(np.array(np.where(dr_sim_new == 1)).T, columns=['Drug1', 'Drug2'])
        di_di_new = pd.DataFrame(np.array(np.where(di_sim_new == 1)).T, columns=['Disease1', 'Disease2'])

        graph_data_llm = {
            ('drug', 'dr_dr', 'drug'): (th.tensor(dr_dr_new['Drug1'].values), th.tensor(dr_dr_new['Drug2'].values)),
            ('disease', 'di_di', 'disease'): (th.tensor(di_di_new['Disease1'].values), th.tensor(di_di_new['Disease2'].values)),
            ('drug', 'dr_di', 'disease'): (th.tensor(dr_di['Drug'].values), th.tensor(dr_di['Disease'].values)),
            ('disease', 'di_dr', 'drug'): (th.tensor(dr_di['Disease'].values), th.tensor(dr_di['Drug'].values)),
        }
        g_llm = dgl.heterograph(graph_data_llm)
        
        dr_feat_llm = np.hstack((dr_sim_new, np.zeros((g_llm.num_nodes('drug'), g_llm.num_nodes('disease')))))
        di_feat_llm = np.hstack((np.zeros((g_llm.num_nodes('disease'), g_llm.num_nodes('drug'))), di_sim_new))
        g_llm.nodes['drug'].data['h'] = th.from_numpy(dr_feat_llm).to(th.float32)
        g_llm.nodes['disease'].data['h'] = th.from_numpy(di_feat_llm).to(th.float32)
    else:
        g_llm = g

    return g, g_llm


#Delete the drug-disease associations which belong to test set from heterogeneous network.
   
def remove_graph(g, test_id):
    """حذف الروابط الخاصة بمجموعة الاختبار"""
    test_drug_id = test_id[:, 0]
    test_dis_id = test_id[:, 1]
    edges_id = g.edge_ids(th.tensor(test_drug_id), th.tensor(test_dis_id), etype=('drug', 'dr_di', 'disease'))
    g = dgl.remove_edges(g, edges_id, etype=('drug', 'dr_di', 'disease'))
    edges_id_rev = g.edge_ids(th.tensor(test_dis_id), th.tensor(test_drug_id), etype=('disease', 'di_dr', 'drug'))
    g = dgl.remove_edges(g, edges_id_rev, etype=('disease', 'di_dr', 'drug'))
    return g


#Generate the node features for the heterogeneous network.
#ضمان مرونة النظام في استقبال الجرافات؛ حيث يمكن للدالة التعامل مع الجرافات سواء مُررت كعناصر منفصلة أو كقائمة مدمجة (List).
def generate_feat(args, g, g_llm=None):
    """توليد ميزات العقد النهائية"""
   
    if isinstance(g, list):
        g_llm = g[1]
        g = g[0]
        
# 1. بصمة Morgan (Fingerprint)
    morgan_tensor = None  # تهيئة آمنة لتجنب NameError

    if args.dr_fingerprint:
        morgan_path = f'../feat/{args.dataset}/Morgan_drug_emb.pkl'
        with open(morgan_path, 'rb') as f:
            morgan_data = pickle.load(f)
        
        # --- التعديل هنا لربط الـ ID بالـ Index ---
        # نقرأ ملف الأدوية لنعرف ترتيب الأسماء (مثل DB0001) الذي يطابق ترتيب نود الجراف
        drug_info = pd.read_csv(f'../data/{args.dataset}/drug.csv')
        drug_list = drug_info['Drug'].values  # أسماء الأدوية بالترتيب الصحيح
        
        fps = []
        for d_id in drug_list:
            if d_id in morgan_data:
                fps.append(morgan_data[d_id])
            else:
                # في حال عدم وجود الدواء نضع أصفاراً
                fps.append(np.zeros(1024))
        
        fps = np.array(fps)
        # ------------------------------------------

        morgan_tensor = th.tensor(fps).to(th.float32).to(args.device)
        dr_feat = th.cat([g.nodes['drug'].data['h'], morgan_tensor], dim=1)
    else:
        dr_feat = g.nodes['drug'].data['h']


        
    # 2. بروتينات الأمراض (لـ Bdataset)
    if args.dis_prot_assoc and args.dataset == 'Bdataset':
        dis_prot = pd.read_csv(f'../data/{args.dataset}/protein_disease.csv')
        num_prot = dis_prot['Protein'].max()
        num_dis = dis_prot['Disease'].max()
        dis_prot_matrix = np.zeros((num_dis+1, num_prot+1))
        dis_prot_matrix[dis_prot['Disease'], dis_prot['Protein']] = 1
        g.nodes['disease'].data['dis_prot_assoc'] = th.from_numpy(dis_prot_matrix).to(th.float32).to(args.device)
        dis_feat = th.cat([g.nodes['disease'].data['h'], g.nodes['disease'].data['dis_prot_assoc']], dim=1)
    else:
        dis_feat = g.nodes['disease'].data['h']
    
   # 3. استخراج الـ Embeddings (دعم شامل لـ Morgan و ChemBERTa و BERT و LLM)
    drug_LLM_emb = None
    disease_LLM_emb = None

    # الشرط الشامل لضمان عدم حدوث خطأ NoneType
   #في الكود القديم: إذا لم يتم تفعيل BERT_emb أو LLM_emb بالصدفة، قد يحاول الكود دمج متغيرات غير موجودة، مما يؤدي لخطأ NoneType.
    if args.feature_type in ['Morgan', 'ChemBERTa', 'BERT', 'LLM'] or args.BERT_emb or args.LLM_emb:
        
        # أ: تحميل BERT للأمراض دائماً (ثابت كمرجع )
        disease_path = f'../feat/{args.dataset}/BERT_disease_emb.pkl'
        with open(disease_path, 'rb') as f:
            di_emb_dict = pickle.load(f)
        disease_LLM_emb = th.tensor(list(di_emb_dict.values())).to(th.float32).to(args.device)

        #  تحديد ميزات الأدوية بناءً على النوع المختارة
        if args.feature_type == 'Morgan':
            
           if morgan_tensor is None:
                raise ValueError(
                    "feature_type='Morgan' يتطلب args.dr_fingerprint=True. "
                    "يرجى التأكد من تفعيل البصمات الكيميائية."
                )
           drug_LLM_emb = morgan_tensor
        
        elif args.feature_type == 'ChemBERTa':
            # تحميل ChemBERTa للأدوية
            with open(f'../feat/{args.dataset}/ChemBERTa_drug_emb.pkl', 'rb') as f:
                dr_emb_dict = pickle.load(f)
            # التأكد من الترتيب الصحيح للأدوية
            drug_LLM_emb = th.tensor([dr_emb_dict[i] for i in sorted(dr_emb_dict.keys())]).to(th.float32).to(args.device)
            
        else:
            # في حال اختيار BERT أو LLM الأصليين
            drug_LLM_emb = g_llm.nodes['drug'].data['h']

 # 4. الإرجاع بناءً على نوع الدمج (Return)
    if args.concatenate_type == 'as_node':
        return {'drug': th.cat([dr_feat, drug_LLM_emb], dim=1),
                'disease': th.cat([dis_feat, disease_LLM_emb], dim=1)}
    
    elif args.concatenate_type == 'none':
        return {'drug': dr_feat, 'disease': dis_feat}
    
    else:
        # إذا كانت القيم None، نعطيها مصفوفة أصفار بنفس أبعاد ميزات الجراف
        # لكي يقرأ المودل الـ shape[1] بنجاح ولا ينهار
        if drug_LLM_emb is None:
            drug_LLM_emb = th.zeros_like(dr_feat)
        if disease_LLM_emb is None:
            disease_LLM_emb = th.zeros_like(dis_feat)

        return {'drug': dr_feat, 
                'disease': dis_feat,
                'drug_LLM': drug_LLM_emb, 
                'disease_LLM': disease_LLM_emb}