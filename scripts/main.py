from mineru_processor import main as mineru_processor_main
from common_data_processor import main as common_data_processor_main
from common_json_clean import main as common_json_clean_main
from triplets_generate_api import main as triplets_generate_api_main
from train import main as train_main
from test_data_generate import main as test_data_generate_main
from test import main as test_main

def run_pipeline():
    print("=== 开始完整流程 ===")

    print("\n📊 阶段0: 数据预处理")
    mineru_processor_main()
    common_data_processor_main()
    common_json_clean_main()

    print("\n📊 阶段1: 数据生成")
    triplets_generate_api_main()

    print("\n🎯 阶段2: 模型训练")
    train_main()

    print("\n🧪 阶段3: 生成测试集")
    test_data_generate_main()


    print("\n📈 阶段4: 测试")
    test_main()

if __name__ == "__main__":
    run_pipeline()