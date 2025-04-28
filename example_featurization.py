import numpy as np
import os
import glob
from rdkit import Chem
from collections.abc import Mapping
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

def read_smi_file(smi_file):
    """
    读取smi文件中的SMILES字符串
    
    参数:
    smi_file: str, smi文件路径
    
    返回:
    list: SMILES字符串列表
    """
    smiles_list = []
    with open(smi_file, 'r') as f:
        for line in f:
            # 假设每行格式为: SMILES\t名称
            smiles = line.strip().split('\t')[0]
            if smiles:  # 确保不是空行
                smiles_list.append(smiles)
    return smiles_list

def extract_mol2vec_features(smiles_list, model_path=None, radius=1):
    """
    使用mol2vec提取分子特征
    
    参数:
    smiles_list: list of str, SMILES字符串列表
    model_path: str, 预训练的word2vec模型路径，如果为None则使用默认路径
    radius: int, Morgan指纹的半径
    
    返回:
    np.array: 分子特征向量
    """
    try:
        # 设置默认模型路径
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'model_300dim.pkl')
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到预训练模型文件: {model_path}\n"
                                  f"请从 https://github.com/samoturk/mol2vec/tree/master/examples/models 下载模型文件")
        
        # 加载预训练模型
        model = word2vec.Word2Vec.load(model_path)
        
        # 将SMILES转换为RDKit分子对象
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        
        # 生成分子句子
        sentences = [MolSentence(mol2alt_sentence(mol, radius)) for mol in mols]
        
        # 将句子转换为向量
        vectors = sentences2vec(sentences, model, unseen='UNK')
        
        # 转换为numpy数组
        if isinstance(vectors[0], DfVec):
            feature_vectors = np.array([vec.vec for vec in vectors])
        else:
            feature_vectors = np.array(vectors)
        
        return feature_vectors
    except Exception as e:
        print(f"错误: {str(e)}")
        return None

def process_smi_files(input_dir, output_dir, model_path=None, radius=1):
    """
    处理指定目录下的所有smi文件
    
    参数:
    input_dir: str, 输入目录路径
    output_dir: str, 输出目录路径
    model_path: str, 预训练模型路径
    radius: int, Morgan指纹的半径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有smi文件
    smi_files = glob.glob(os.path.join(input_dir, '*.smi'))
    
    for smi_file in smi_files:
        try:
            print(f"处理文件: {smi_file}")
            
            # 读取SMILES
            smiles_list = read_smi_file(smi_file)
            
            # 提取特征
            features = extract_mol2vec_features(smiles_list, model_path, radius)
            
            if features is not None:
                # 生成输出文件名
                base_name = os.path.splitext(os.path.basename(smi_file))[0]
                output_file = os.path.join(output_dir, f"{base_name}_features.npy")
                
                # 保存特征
                np.save(output_file, features)
                print(f"特征已保存到: {output_file}")
                print(f"特征向量形状: {features.shape}")
            else:
                print(f"处理文件 {smi_file} 失败")
                
        except Exception as e:
            print(f"处理文件 {smi_file} 时出错: {str(e)}")

if __name__ == "__main__":
    # 设置输入和输出目录
    input_dir = "input_smi"  # 存放smi文件的目录
    output_dir = "output_features"  # 存放特征文件的目录
    
    # 处理所有smi文件
    process_smi_files(input_dir, output_dir) 