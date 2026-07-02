"""
训练时间跟踪和统计模块
为不同的生成模型提供统一的训练时间记录功能
"""

import time
import json
import os
from datetime import datetime
from typing import Dict, Optional


class TrainingTimeTracker:
    """
    训练时间跟踪器
    记录模型训练的各个阶段耗时
    """
    
    def __init__(self, model_name: str):
        """
        初始化跟踪器
        
        Args:
            model_name: 模型名称 (GMM-FlowMatching, FlowMatching, DDPM, DCGAN, WGAN-CP, WGAN-GP)
        """
        self.model_name = model_name
        self.start_time = None
        self.end_time = None
        self.phase_times = {}  # 存储各阶段的耗时
        self.total_time = None
        
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        
    def end_phase(self, phase_name: str):
        """
        结束一个阶段的计时
        
        Args:
            phase_name: 阶段名称 (如 'model_training', 'data_generation', 'classification')
        """
        if self.start_time is None:
            print(f"警告：请先调用 start() 方法")
            return
        
        current_time = time.time()
        phase_time = current_time - self.start_time
        self.phase_times[phase_name] = phase_time
        
        # 打印阶段耗时
        minutes = int(phase_time // 60)
        seconds = phase_time % 60
        print(f"  ✓ {phase_name} 耗时: {minutes}分{seconds:.2f}秒 ({phase_time:.2f}秒)")
    
    def end(self):
        """结束总计时"""
        if self.start_time is None:
            print(f"警告：请先调用 start() 方法")
            return
        
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
    
    def get_total_time(self) -> float:
        """获取总耗时（秒）"""
        if self.total_time is None:
            if self.start_time is None:
                return 0
            self.total_time = time.time() - self.start_time
        return self.total_time
    
    def get_total_time_str(self) -> str:
        """获取格式化的总耗时字符串"""
        total = self.get_total_time()
        minutes = int(total // 60)
        seconds = total % 60
        hours = minutes // 60
        minutes = minutes % 60
        
        if hours > 0:
            return f"{hours}小时{minutes}分{seconds:.2f}秒"
        else:
            return f"{minutes}分{seconds:.2f}秒"
    
    def print_summary(self):
        """打印总结信息"""
        print("\n" + "="*80)
        print(f"【{self.model_name} 训练时间统计】")
        print("="*80)
        
        if self.phase_times:
            print("\n► 各阶段耗时：")
            for phase_name, phase_time in self.phase_times.items():
                minutes = int(phase_time // 60)
                seconds = phase_time % 60
                percentage = (phase_time / self.get_total_time() * 100) if self.get_total_time() > 0 else 0
                print(f"  • {phase_name:<30} {minutes:>3}分{seconds:>6.2f}秒 ({percentage:>5.1f}%)")
        
        print(f"\n► 总耗时: {self.get_total_time_str()}")
        print("="*80 + "\n")
    
    def save_json(self, save_path: str = "./results/training_times/"):
        """
        保存为JSON文件
        
        Args:
            save_path: 保存路径
        """
        os.makedirs(save_path, exist_ok=True)
        
        # 准备数据
        data = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': self.get_total_time(),
            'total_time_formatted': self.get_total_time_str(),
            'phase_times': {}
        }
        
        # 添加各阶段耗时
        for phase_name, phase_time in self.phase_times.items():
            minutes = int(phase_time // 60)
            seconds = phase_time % 60
            data['phase_times'][phase_name] = {
                'seconds': phase_time,
                'formatted': f"{minutes}分{seconds:.2f}秒"
            }
        
        # 保存为JSON
        filename = f"{save_path}training_time_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 训练时间记录已保存: {filename}")
        
        return filename


def aggregate_training_times(results_dir: str = "./results/training_times/") -> Dict:
    """
    聚合所有训练时间数据，生成对比报告
    
    Args:
        results_dir: 训练时间结果目录
    
    Returns:
        包含所有模型训练时间的字典
    """
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        return {}
    
    model_times = {}
    
    # 扫描所有JSON文件
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename.startswith('training_time_'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    model_name = data.get('model_name')
                    total_time = data.get('total_time_seconds', 0)
                    
                    # 保存最新的记录
                    if model_name not in model_times or data.get('timestamp') > model_times[model_name].get('timestamp', ''):
                        model_times[model_name] = data
            except Exception as e:
                print(f"无法读取文件 {filename}: {e}")
    
    return model_times


def generate_training_time_report(results_dir: str = "./results/training_times/"):
    """
    生成训练时间对比报告
    
    Args:
        results_dir: 训练时间结果目录
    """
    model_times = aggregate_training_times(results_dir)
    
    if not model_times:
        print("没有找到训练时间数据")
        return
    
    # 按模型名排序
    sorted_models = sorted(model_times.items(), 
                          key=lambda x: x[1].get('total_time_seconds', 0))
    
    # 找出最快的模型作为基准
    fastest_time = sorted_models[0][1]['total_time_seconds']
    
    # 生成报告
    report_lines = []
    report_lines.append("\n" + "="*100)
    report_lines.append("【生成模型训练时间对比分析】")
    report_lines.append("="*100)
    report_lines.append("")
    report_lines.append(f"{'模型名称':<25} {'训练时间':<20} {'相对时间(倍数)':<20} {'时间戳':<30}")
    report_lines.append("-"*100)
    
    for model_name, data in sorted_models:
        total_time = data['total_time_seconds']
        formatted_time = data['total_time_formatted']
        relative_time = total_time / fastest_time if fastest_time > 0 else 1.0
        timestamp = data.get('timestamp', 'N/A')
        
        report_lines.append(f"{model_name:<25} {formatted_time:<20} {relative_time:>6.2f}x{'':<12} {timestamp:<30}")
    
    report_lines.append("="*100)
    report_lines.append(f"\n► 最快模型: {sorted_models[0][0]} ({sorted_models[0][1]['total_time_formatted']})")
    report_lines.append(f"► 最慢模型: {sorted_models[-1][0]} ({sorted_models[-1][1]['total_time_formatted']})")
    
    # 计算平均时间
    avg_time = sum(data['total_time_seconds'] for _, data in sorted_models) / len(sorted_models)
    avg_minutes = int(avg_time // 60)
    avg_seconds = avg_time % 60
    report_lines.append(f"► 平均训练时间: {avg_minutes}分{avg_seconds:.2f}秒")
    
    report_lines.append("\n")
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # 保存报告
    os.makedirs(results_dir, exist_ok=True)
    report_file = os.path.join(results_dir, f"training_time_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ 对比报告已保存: {report_file}\n")


if __name__ == "__main__":
    # 示例用法
    print("训练时间跟踪器示例")
    
    # 创建跟踪器
    tracker = TrainingTimeTracker("GMM-FlowMatching")
    tracker.start()
    
    # 模拟各阶段
    time.sleep(1)
    tracker.end_phase("model_training")
    
    time.sleep(0.5)
    tracker.end_phase("data_generation")
    
    tracker.end()
    tracker.print_summary()
