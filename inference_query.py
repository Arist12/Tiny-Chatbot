import torch
import pandas as pd
import json
import re
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import requests
import os
import argparse
from dotenv import load_dotenv

# 加载环境变量（用于API密钥）
load_dotenv()

# ========== 添加命令行参数解析 ==========
def parse_args():
    parser = argparse.ArgumentParser(description='MBTI Personality Analyzer')
    parser.add_argument('--input', type=str, help='Input text file path')
    parser.add_argument('--api-only', action='store_true', help='Return JSON output for API use')
    return parser.parse_args()

# ========== BERT分类器定义 ==========
class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        return self.classifier(cls_output)

# ========== 文本预处理和预测函数 ==========
def preprocess_text(text):
    # 简单清理文本
    text = re.sub(r'https?://\S+', '', text)  # 移除链接
    text = re.sub(r'\s+', ' ', text).strip()  # 规范化空格
    return text

def predict_mbti(text, model, tokenizer, label2mbti, device):
    # 预处理文本
    text = preprocess_text(text)
    
    # 使用tokenizer处理文本
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    # 将数据移动到正确的设备
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 进行预测
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
    
    # 获取完整的概率分布
    probabilities = probs[0].cpu().numpy()
    
    # 返回预测的MBTI类型和置信度
    mbti_type = label2mbti[pred_class]
    confidence = float(probs[0][pred_class])
    
    # 创建一个包含所有MBTI类型及其概率的字典
    all_probs = {label2mbti[i]: float(probabilities[i]) for i in range(len(label2mbti))}
    
    return mbti_type, confidence, all_probs

# ========== 向LLM发送请求的函数 ==========
def query_llm(prompt, api_key, model="gpt-3.5-turbo"):
    """向LLM发送请求并获取响应"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# ========== 主函数 ==========
def main():
    args = parse_args()
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.api_only:
        print(f"Using device: {device}")
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 加载label到MBTI的映射
    with open("mbti2label.json", "r") as f:
        mbti2label = json.load(f)
    
    # 创建反向映射
    label2mbti = {v: k for k, v in mbti2label.items()}
    
    # 初始化模型
    num_classes = len(mbti2label)
    model = BERTClassifier(num_classes).to(device)
    
    # 加载训练好的模型权重
    model.load_state_dict(torch.load("best_bert_mbti_classifier.pt", map_location=device))
    
    # 设置prompt模板
    prompt_template = """
    作为一位MBTI专家，我需要你扮演{mbti_type}类型的角色来回应用户的问题。
    
    MBTI分析结果: {mbti_type}（置信度: {confidence:.2f}）
    
    {mbti_type}类型的主要特征：
    - 如果是INTJ: 独立思考，战略性思维，注重效率与逻辑
    - 如果是INTP: 好奇心强，理论分析能力强，喜欢解决复杂问题
    - 如果是ENTJ: 果断，有领导力，注重目标与效率
    - 如果是ENTP: 创新，辩论能力强，喜欢挑战现状
    
    
    用户原文：
    {user_text}
    
    请以{mbti_type}类型的视角，用友好的口吻回应上述内容，体现这种性格类型的思维方式与表达特点。
    """
    
    # 如果提供了输入文件，则从文件读取文本
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            user_text = f.read()
        
        # 预测MBTI
        mbti_type, confidence, all_probs = predict_mbti(user_text, model, tokenizer, label2mbti, device)
        
        # 填充prompt模板
        prompt = prompt_template.format(
            mbti_type=mbti_type,
            confidence=confidence,
            user_text=user_text
        )
        
        # 向LLM发送请求
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            llm_response = query_llm(prompt, api_key)
            
            if args.api_only:
                # 返回JSON格式的结果
                result = {
                    "mbti_type": mbti_type,
                    "confidence": confidence,
                    "all_probabilities": all_probs,
                    "explanation": llm_response
                }
                print(json.dumps(result))
            else:
                print(f"\n预测的MBTI类型: {mbti_type} (置信度: {confidence:.4f})")
                print("所有MBTI类型的概率分布:")
                for mbti, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {mbti}: {prob:.4f}")
                print("\n===== LLM 响应 =====")
                print(llm_response)
                print("=====================\n")
        else:
            if args.api_only:
                result = {
                    "mbti_type": mbti_type,
                    "confidence": confidence,
                    "all_probabilities": all_probs,
                    "error": "API key not found"
                }
                print(json.dumps(result))
            else:
                print("\n未找到API密钥，无法向LLM发送请求。")
                print("请设置OPENAI_API_KEY环境变量或在.env文件中添加。")
    else:
        # 原有的交互式模式
        if not args.api_only:
            print("请输入文本进行MBTI分析（输入'quit'退出）：")
            while True:
                user_text = input("> ")
                if user_text.lower() == 'quit':
                    break
                
                # 预测MBTI
                mbti_type, confidence, all_probs = predict_mbti(user_text, model, tokenizer, label2mbti, device)
                
                print(f"\n预测的MBTI类型: {mbti_type} (置信度: {confidence:.4f})")
                print("所有MBTI类型的概率分布:")
                for mbti, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {mbti}: {prob:.4f}")
                
                # 填充prompt模板
                prompt = prompt_template.format(
                    mbti_type=mbti_type,
                    confidence=confidence,
                    user_text=user_text
                )
                
                # 向LLM发送请求
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    print("\n正在向LLM发送请求...")
                    llm_response = query_llm(prompt, api_key)
                    print("\n===== LLM 响应 =====")
                    print(llm_response)
                    print("=====================\n")
                else:
                    print("\n未找到API密钥，无法向LLM发送请求。")
                    print("请设置OPENAI_API_KEY环境变量或在.env文件中添加。")
                print("\n请输入新的文本进行分析（输入'quit'退出）：")

if __name__ == "__main__":
    main()