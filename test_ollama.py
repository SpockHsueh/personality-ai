"""
測試 Ollama 客戶端模組
基於官方 ollama-python 文件
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.ollama_client import OllamaClient, get_ollama_client, simple_chat, simple_generate
import ollama
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_basic_connection():
    """測試基本連接"""
    print("=== 測試 Ollama 連接 ===")
    client = OllamaClient()
    
    # 測試連接
    is_connected = client.test_connection()
    print(f"連接狀態: {'✅ 成功' if is_connected else '❌ 失敗'}")
    
    if is_connected:
        # 列出可用模型
        models = client.list_models()
        print(f"\n可用模型數量: {len(models)}")
        for model in models:
            name = model.get('name', 'N/A')
            size = model.get('size', 0)
            # 轉換大小為更友善的格式
            if size > 1e9:
                size_str = f"{size/1e9:.1f}GB"
            elif size > 1e6:
                size_str = f"{size/1e6:.1f}MB"
            else:
                size_str = f"{size}B"
            print(f"  - {name} (大小: {size_str})")
    
    return is_connected

def test_simple_functions():
    """測試全域簡單函數"""
    print("\n=== 測試全域簡單函數 ===")
    
    # 測試 simple_chat
    print("測試 simple_chat:")
    prompt = "請用一句話介紹你自己"
    print(f"輸入: {prompt}")
    
    response = simple_chat(prompt)
    if response:
        print(f"回應: {response[:100]}...")  # 只顯示前100字
        return True
    else:
        print("❌ simple_chat 失敗")
        return False

def test_client_generation():
    """測試客戶端生成"""
    print("\n=== 測試客戶端文字生成 ===")
    client = get_ollama_client()
    
    prompt = "請簡單介紹台灣的特色"
    print(f"輸入: {prompt}")
    
    response = client.generate_response(prompt, temperature=0.5, max_tokens=100)
    
    if response:
        print(f"回應: {response}")
        return True
    else:
        print("❌ 生成失敗")
        return False

def test_personality_generation():
    """測試人格化生成"""
    print("\n=== 測試人格化生成 ===")
    client = get_ollama_client()
    
    # 定義一個簡單的人格
    personality_prompt = """你是一個熱情外向的ENFP類型人格，具有以下特徵：
- 充滿熱情和創意
- 喜歡鼓勵他人
- 說話生動有趣，經常使用比喻
- 對新可能性感到興奮

請以這種人格風格回答問題。"""
    
    prompt = "今天天氣很好，你有什麼建議嗎？"
    print(f"輸入: {prompt}")
    print("人格設定: ENFP 熱情外向型")
    
    response = client.generate_with_personality(prompt, personality_prompt)
    
    if response:
        print(f"個性化回應: {response}")
        return True
    else:
        print("❌ 個性化生成失敗")
        return False

def test_chat_history():
    """測試對話歷史功能"""
    print("\n=== 測試對話歷史 ===")
    client = get_ollama_client()
    
    # 模擬對話歷史
    messages = [
        {'role': 'system', 'content': '你是一個友善的助手，喜歡用簡潔的方式回答問題。'},
        {'role': 'user', 'content': '你好！我叫小明。'},
        {'role': 'assistant', 'content': '你好小明！很高興認識你。'},
        {'role': 'user', 'content': '你還記得我的名字嗎？'}
    ]
    
    print("對話歷史:")
    for msg in messages[-2:]:  # 只顯示最後兩條
        role = msg['role']
        content = msg['content']
        print(f"  {role}: {content}")
    
    response = client.chat_with_history(messages)
    
    if response:
        print(f"回應: {response}")
        return True
    else:
        print("❌ 對話歷史測試失敗")
        return False

def test_model_availability():
    """測試模型可用性"""
    print("\n=== 測試模型可用性 ===")
    client = get_ollama_client()
    
    # 檢查預設模型是否可用
    models = client.list_models()
    model_names = [m.get('name', '') for m in models]
    
    print(f"預設模型: {client.model_name}")
    if client.model_name in model_names:
        print("✅ 預設模型可用")
        return True
    else:
        print("⚠️ 預設模型不可用，可用模型:")
        for name in model_names:
            print(f"  - {name}")
        
        # 嘗試使用第一個可用模型
        if model_names and any(name for name in model_names if name):
            test_model = next(name for name in model_names if name)
            print(f"\n嘗試使用模型: {test_model}")
            response = client.generate_response("Hello", model=test_model)
            if response:
                print(f"✅ 模型 {test_model} 工作正常")
                return True
        
        return False

def main():
    """主測試函數"""
    print("🚀 開始測試 Ollama 客戶端模組 (官方 API 版本)\n")
    
    # 測試順序
    tests = [
        ("基本連接", test_basic_connection),
        ("模型可用性", test_model_availability),
        ("全域簡單函數", test_simple_functions),
        ("客戶端生成", test_client_generation),
        ("人格化生成", test_personality_generation),
        ("對話歷史", test_chat_history)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*50}")
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 測試出錯: {e}")
            results.append((test_name, False))
    
    # 總結結果
    print(f"\n{'='*50}")
    print("📊 測試結果總結:")
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"  {test_name}: {status}")
    
    success_count = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    print(f"\n總計: {success_count}/{total_tests} 測試通過")
    
    if success_count == total_tests:
        print("🎉 所有測試通過！可以進行下一步開發。")
    elif success_count > 0:
        print("⚠️ 部分測試通過，可以繼續開發，但需要注意失敗的項目。")
    else:
        print("💥 所有測試失敗，請檢查 Ollama 服務和模型安裝。")

if __name__ == "__main__":
    main()