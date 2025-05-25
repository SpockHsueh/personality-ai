"""
Ollama 客戶端模組
基於官方 ollama-python 文件的正確實作
"""
import ollama
from ollama import Client, ResponseError
import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import logging

# 載入環境變數
load_dotenv()

logger = logging.getLogger(__name__)

class OllamaClient:
    """Ollama 客戶端類別"""
    
    def __init__(self, host: str = None, model_name: str = None):
        """
        初始化 Ollama 客戶端
        
        Args:
            host: Ollama 服務地址
            model_name: 預設模型名稱
        """
        self.host = host or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.model_name = model_name or os.getenv('MODEL_NAME', 'gemma3:27b')
        
        # 建立客戶端實例
        self.client = Client(host=self.host)
        
        logger.info(f"初始化 Ollama 客戶端: {self.host}, 模型: {self.model_name}")
    
    def test_connection(self) -> bool:
        """
        測試 Ollama 連接
        
        Returns:
            bool: 連接是否成功
        """
        try:
            models = self.client.list()
            model_count = len(models.get('models', []))
            logger.info(f"成功連接到 Ollama，可用模型數量: {model_count}")
            return True
        except Exception as e:
            logger.error(f"連接 Ollama 失敗: {e}")
            return False
    
    def list_models(self) -> List[Dict]:
        """
        獲取可用模型列表
        
        Returns:
            List[Dict]: 模型資訊列表
        """
        try:
            response = self.client.list()
            models = response.get('models', [])
            
            # 轉換 Pydantic 對象為字典格式
            processed_models = []
            for model in models:
                if hasattr(model, 'model'):  # Pydantic 對象
                    processed_models.append({
                        'name': model.model,
                        'size': getattr(model, 'size', 0),
                        'modified_at': getattr(model, 'modified_at', None),
                        'digest': getattr(model, 'digest', ''),
                        'details': getattr(model, 'details', {})
                    })
                else:  # 已經是字典
                    processed_models.append(model)
            
            return processed_models
            
        except Exception as e:
            logger.error(f"獲取模型列表失敗: {e}")
            return []
    
    def generate_response(self, 
                         prompt: str, 
                         model: str = None,
                         system_prompt: str = None,
                         temperature: float = 0.7,
                         max_tokens: int = 1000) -> Optional[str]:
        """
        生成回應
        
        Args:
            prompt: 使用者輸入
            model: 指定模型（可選）
            system_prompt: 系統提示（可選）
            temperature: 生成溫度
            max_tokens: 最大 token 數
            
        Returns:
            Optional[str]: 生成的回應
        """
        model = model or self.model_name
        
        try:
            # 建立訊息陣列
            messages = []
            
            # 添加系統提示
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            
            # 添加使用者輸入
            messages.append({
                'role': 'user',
                'content': prompt
            })
            
            # 調用 ollama.chat API
            response = self.client.chat(
                model=model,
                messages=messages,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens
                }
            )
            
            # 根據官方文件，回應格式是 response['message']['content']
            return response['message']['content']
            
        except ResponseError as e:
            logger.error(f"Ollama 回應錯誤: {e.error} (狀態碼: {e.status_code})")
            return None
        except Exception as e:
            logger.error(f"生成回應失敗: {e}")
            return None
    
    def generate_with_personality(self, 
                                 prompt: str,
                                 personality_prompt: str,
                                 model: str = None) -> Optional[str]:
        """
        使用特定人格生成回應
        
        Args:
            prompt: 使用者輸入
            personality_prompt: 人格化提示
            model: 指定模型
            
        Returns:
            Optional[str]: 個性化回應
        """
        return self.generate_response(
            prompt=prompt,
            system_prompt=personality_prompt,
            model=model,
            temperature=0.8  # 提高創意性
        )
    
    def generate_simple(self, prompt: str, model: str = None) -> Optional[str]:
        """
        使用 generate API 生成簡單回應
        
        Args:
            prompt: 輸入提示
            model: 指定模型
            
        Returns:
            Optional[str]: 生成的回應
        """
        model = model or self.model_name
        
        try:
            response = self.client.generate(
                model=model,
                prompt=prompt
            )
            return response['response']
            
        except ResponseError as e:
            logger.error(f"Ollama 回應錯誤: {e.error} (狀態碼: {e.status_code})")
            return None
        except Exception as e:
            logger.error(f"生成回應失敗: {e}")
            return None
    
    def chat_with_history(self,
                         messages: List[Dict[str, str]],
                         model: str = None,
                         temperature: float = 0.7) -> Optional[str]:
        """
        使用對話歷史生成回應
        
        Args:
            messages: 對話歷史，格式: [{'role': 'user', 'content': '...'}]
            model: 指定模型
            temperature: 生成溫度
            
        Returns:
            Optional[str]: 生成的回應
        """
        model = model or self.model_name
        
        try:
            response = self.client.chat(
                model=model,
                messages=messages,
                options={'temperature': temperature}
            )
            
            return response['message']['content']
            
        except ResponseError as e:
            logger.error(f"對話生成錯誤: {e.error}")
            return None
        except Exception as e:
            logger.error(f"對話生成失敗: {e}")
            return None
    
    def pull_model_if_needed(self, model: str) -> bool:
        """
        檢查並下載模型（如果需要）
        
        Args:
            model: 模型名稱
            
        Returns:
            bool: 是否成功
        """
        try:
            # 檢查模型是否已存在
            models = self.list_models()
            model_names = [m.get('name', '') for m in models]
            
            if model in model_names:
                logger.info(f"模型 {model} 已存在")
                return True
            
            # 下載模型
            logger.info(f"正在下載模型 {model}...")
            self.client.pull(model)
            logger.info(f"模型 {model} 下載完成")
            return True
            
        except Exception as e:
            logger.error(f"下載模型失敗: {e}")
            return False

# 全域函數 - 直接使用 ollama 模組
def simple_chat(message: str, model: str = None) -> Optional[str]:
    """
    簡單對話函數
    
    Args:
        message: 訊息內容
        model: 模型名稱 (None 時使用環境變數或預設值)
        
    Returns:
        Optional[str]: 回應內容
    """
    if model is None:
        model = os.getenv('MODEL_NAME', 'gemma3:1b')  # 改為小模型預設
    
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': message}]
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"簡單對話失敗: {e}")
        return None

# 單例模式的全域客戶端
_ollama_client = None

def get_ollama_client() -> OllamaClient:
    """獲取全域 Ollama 客戶端實例"""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client