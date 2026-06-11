import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# General Arguments
parser.add_argument('-id', '--device_id', default=None, type=str,
                    help='Set the device (GPU ids).')
parser.add_argument('-da', '--dataset', type=str,
                    default="Cdataset",#اضافة داتاسيت افتراضيه 
                    choices=['Bdataset', 'Cdataset', 'Fdataset', 'Rdataset'],
                    help='Set the data set for training.')

#تمت إضافة ميزة (,LLM,BERT) في ملف تحميل البيانات ولم تُضف في ملف تعريف المتغيرات.

parser.add_argument('--BERT_emb', action='store_true', default=False,
                    help='Use BERT embeddings .')
parser.add_argument('--LLM_emb', action='store_true', default=False,
                    help='Use LLM embeddings.')

#--BERT_emb, --LLM_emb اضفتهم

parser.add_argument('-sp', '--saved_path', type=str,
                    help='Path to save training results', default='result')
parser.add_argument('-se', '--seed', default=0, type=int,
                    help='Global random seed')
# Training Arguments
parser.add_argument('-fo', '--nfold', default=5, type=int,
                    help='The number of k in K-folds Validation')
parser.add_argument('-ep', '--epoch', default=5000, type=int,
                    help='Number of epochs for training')
parser.add_argument('-lr', '--learning_rate', default=0.01, type=float,
                    help='learning rate to use')
parser.add_argument('-wd', '--weight_decay', default=0, type=float,
                    help='weight decay to use')
parser.add_argument('-pa', '--patience', default=100, type=int,
                    help='Early Stopping argument')
# Model Arguments
parser.add_argument('-ft', '--feature_type', default='concat', type=str,
                    choices = ['BERT', 'LLM', 'Morgan','ChemBERTa'],#اضافة  'Morgan','ChemBERTa'
                    help='The type of feature used in the model')
parser.add_argument('-ct', '--concatenate_type', default='graph_ae', type=str,
                    choices = ['graph_graph', 'graph_ae', 'cross_graph', 'as_node', 'none'],
                    help='The type of concatenation in the model')
parser.add_argument('-hf', '--hidden_feats', default=128, type=int,
                    help='The dimension of hidden tensor in the model')
parser.add_argument('-dp', '--dropout', default=0.4, type=float,
                    help='The rate of dropout layer')
#for model_gat_final

parser.add_argument('--num_heads', default=2, type=int,
                    help='Number of GAT attention heads')

parser.add_argument('--clip_grad', default=1.0, type=float,
                    help='Gradient clipping for stable GAT training')

parser.add_argument('--leak_safe_mode', action='store_true',
                    help='Recompute similarity matrices per fold')

#تمت اضافتهم للجزء الخاص باستخدام ال SMILES 
parser.add_argument('--drug_in_feats', default=1024, type=int, help='Input dimension for drug features')
parser.add_argument('--disease_in_feats', default=768, type=int, help='Input dimension for disease features')

parser.add_argument(
    '--exp_name',
    default='GAT',
    type=str
)
parser.add_argument(
    '--hidden_ratio',
    type=float,
    default=0.10,
    help='Fraction of positive links to hide before training.'
)

parser.add_argument(
    '--threshold',
    type=float,
    default=0.5,
    help='Recovery threshold.'
)

args = parser.parse_args()
args.saved_path = os.path.join('../result', args.exp_name,
                    args.dataset+'_'+args.concatenate_type+'_'+args.feature_type+'_'+str(args.epoch) \
                    +'_'+str(args.dropout)+'_'+str(args.hidden_feats),
                    str(args.seed))
args.dr_fingerprint = True
args.dis_prot_assoc = True
if args.feature_type == 'BERT':
    args.BERT_emb, args.LLM_emb = True, False
elif args.feature_type == 'LLM':
    args.BERT_emb, args.LLM_emb = False, True

elif args.feature_type == 'Morgan':
    args.BERT_emb, args.LLM_emb = False, False # لن نستخدم ملفات النصوص
    args.dr_fingerprint = True # سنستخدم البصمات الكيميائية
    
    # تأكد من أن الأبعاد موحدة للجرافين لكي لا ينهار الموديل
    args.drug_in_feats = 1024

 
elif args.feature_type == 'ChemBERTa':
    args.BERT_emb = False
    args.LLM_emb = False
    args.dr_fingerprint = False 
    args.drug_in_feats = 768  # أبعاد ChemBERTa المشهورة