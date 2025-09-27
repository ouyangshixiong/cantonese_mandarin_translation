#!/usr/bin/env python3
"""
测试FP8模型在长句翻译上的表现
"""
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from inference.translator import CantoneseTranslator

def test_long_sentences():
    """测试长句翻译"""
    print("🚀 开始FP8模型长句翻译测试...")
    
    # 初始化翻译器
    translator = CantoneseTranslator(
        model_name="Tencent-Hunyuan/Hunyuan-MT-7B-fp8",
        framework="pytorch",
        device="auto"
    )
    
    # 10个长句测试用例
    test_cases = [
        # 长句1: 日常生活描述
        "我今日朝早去咗茶樓飲茶，見到好多老朋友，大家傾咗好多往事，講起以前一齊做嘢嘅日子，真係好懷念嗰段時間。",
        
        # 长句2: 工作场景
        "老闆今日開會嘅時候講咗好多關於公司未來發展嘅大計，話要擴展業務去海外市場，仲話要請多啲人嚟幫手做嘢。",
        
        # 长句3: 情感表达
        "我覺得人生中最重要嘅唔係賺到幾多錢，而係搵到一個真正懂你嘅人，可以同你一齊分享生活中嘅喜怒哀樂。",
        
        # 长句4: 文化描述
        "粵語文化真係好博大精深，有好多古老嘅諺語同歇後語，每一句都包含住先輩們嘅智慧同人生哲理。",
        
        # 长句5: 家庭场景
        "阿媽今日煮咗我最鍾意食嘅紅燒肉，仲整咗個老火湯，話咁樣對身體好，叫我多喝啲，真係感受到屋企人嘅關懷同溫暖。",
        
        # 长句6: 朋友聚会
        "琴晚同班朋友去咗唱K，大家都玩得好開心，有人唱咗經典粵語歌，有人跳咗舞，整個晚上都充滿住歡樂同笑聲。",
        
        # 长句7: 天气描述
        "今日嘅天氣真係幾好，陽光普照，微風輕輕吹過，啲樹葉沙沙作響，行喺街上感覺特別舒服，令人心情愉快。",
        
        # 长句8: 购物经历
        "我頭先去咗商場買嘢，見到好多新款嘅電子產品，有最新款嘅手機同電腦，科技真係日新月異，發展得好快。",
        
        # 长句9: 节日庆祝
        "就嚟到中秋節喇，每年呢個時候屋企人都會聚埋一齊賞月食月餅，仲會整啲應節嘅小食，大家團團圓圓好溫馨。",
        
        # 长句10: 旅行回忆
        "記得去年去咗旅行，去咗一個好靚嘅海邊小鎮，嗰度嘅風景如畫，海水清澈見底，沙灘潔白如雪，真係一個令人難忘嘅地方。"
    ]
    
    print(f"📋 共準備了 {len(test_cases)} 個長句測試用例")
    print("=" * 80)
    
    results = []
    total_time = 0
    total_chars = 0
    
    for i, cantonese_text in enumerate(test_cases, 1):
        print(f"\n🔍 測試 {i}/10:")
        print(f"粵語原文: {cantonese_text}")
        print(f"字符數: {len(cantonese_text)}")
        
        # 記錄開始時間
        start_time = time.time()
        
        try:
            # 執行翻譯
            mandarin_result = translator.translate(
                cantonese_text, 
                source_lang="cantonese", 
                target_lang="mandarin"
            )
            
            # 計算時間
            inference_time = time.time() - start_time
            total_time += inference_time
            total_chars += len(cantonese_text)
            
            print(f"普通話翻譯: {mandarin_result}")
            print(f"推理時間: {inference_time:.3f}秒")
            print(f"字符速度: {inference_time/len(cantonese_text):.3f}秒/字符")
            
            results.append({
                'id': i,
                'cantonese': cantonese_text,
                'mandarin': mandarin_result,
                'time': inference_time,
                'char_count': len(cantonese_text),
                'char_speed': inference_time/len(cantonese_text)
            })
            
        except Exception as e:
            print(f"❌ 翻譯失敗: {str(e)}")
            results.append({
                'id': i,
                'cantonese': cantonese_text,
                'mandarin': f"翻譯失敗: {str(e)}",
                'time': time.time() - start_time,
                'char_count': len(cantonese_text),
                'char_speed': 0
            })
        
        print("-" * 80)
    
    # 統計結果
    print(f"\n📊 長句翻譯測試統計:")
    print(f"總測試句數: {len(results)}")
    print(f"成功翻譯: {len([r for r in results if '失敗' not in r['mandarin']])}")
    print(f"失敗翻譯: {len([r for r in results if '失敗' in r['mandarin']])}")
    print(f"總推理時間: {total_time:.3f}秒")
    print(f"平均推理時間: {total_time/len(results):.3f}秒")
    print(f"總字符數: {total_chars}")
    print(f"平均字符速度: {total_time/total_chars:.3f}秒/字符")
    
    # 找出最快和最慢的翻譯
    successful_results = [r for r in results if '失敗' not in r['mandarin']]
    if successful_results:
        fastest = min(successful_results, key=lambda x: x['time'])
        slowest = max(successful_results, key=lambda x: x['time'])
        
        print(f"\n⚡ 最快翻譯: 第{fastest['id']}句 - {fastest['time']:.3f}秒")
        print(f"🐌 最慢翻譯: 第{slowest['id']}句 - {slowest['time']:.3f}秒")
    
    return results

if __name__ == "__main__":
    import signal
    
    def timeout_handler(signum, frame):
        print("\n⏰ 测试超时，但继续运行...")
        
    # 设置超时为10分钟
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(600)  # 10分钟
    
    try:
        test_long_sentences()
    finally:
        signal.alarm(0)  # 取消超时