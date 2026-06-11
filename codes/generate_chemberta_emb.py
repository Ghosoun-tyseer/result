import pandas as pd
import torch as th
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle
import os

def generate_chemberta_embeddings(input_csv, output_pkl):
    # 1. إعداد النموذج (استخدام نسخة ChemBERTa-768 المفتوحة)
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # 2. قراءة ملف SMILES
    df = pd.read_csv(input_csv)
    # التأكد من وجود عمود SMILES
    if 'SMILES' not in df.columns:
        raise ValueError("الملف يجب أن يحتوي على عمود باسم SMILES")

    embeddings_dict = {}

    print("جاري تحويل الـ SMILES إلى إيمبدنج...")
    with th.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df)):
            smiles = str(row['SMILES'])
            drug_id = row['ID'] # نستخدم الـ ID كـ Key
            
            # تحويل النص إلى Tokens
            inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)
            # استخراج الميزات من النموذج
            outputs = model(**inputs)
            # نأخذ المتوسط (Mean Pooling) لتمثيل الجزيء بالكامل في ناقل واحد
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            embeddings_dict[drug_id] = emb

    # 3. حفظ النتيجة في مجلد feat
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    
    print(f"تم بنجاح! الملف محفوظ في: {output_pkl}")

if __name__ == "__main__":
    # تأكد من المسار الصحيح لملفك
    generate_chemberta_embeddings('../data/Cdataset/drug.csv', '../feat/Cdataset/ChemBERTa_drug_emb.pkl')