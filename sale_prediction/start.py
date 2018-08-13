import pandas as pd
import numpy as np
from feature_engineering import DataProcess
from model_fusion import Blend
from models import Models


def load_data():
    store = pd.read_csv('input/store.csv')
    train_org = pd.read_csv('input/train.csv', dtype={'StateHoliday': np.string_})
    test_org = pd.read_csv('input/test.csv', dtype={'StateHoliday': np.string_})
    train = pd.merge(train_org, store, on='Store', how='left')
    test = pd.merge(test_org, store, on='Store', how='left')
    features = test.columns.tolist()
    numberics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    features_numberic = test.select_dtypes(include=numberics).columns.tolist()
    feature_non_numberic = [f for f in features if f not in features_numberic]
    return (train, test, features, feature_non_numberic)


def get_cleaned_data():
    (train, test, features, feature_non_numberic) = load_data()
    dataprocessor = DataProcess(train, test, features, feature_non_numberic)
    dataprocessor.process_data()
    dataprocessor.fillna_()
    dataprocessor.encoder()
    dataprocessor.std_scaler()
    y_train = pd.DataFrame()
    y_train['Sales'] = pd.Series(dataprocessor.train_df.Sales)
    x_train = dataprocessor.train_df.drop('Sales', 1, inplace=False)
    x_test = dataprocessor.test_df
    print('开始生成数据'.center(50, '*'))
    x_train.to_csv('cleaned_data/cleaned_x_train.csv')
    print('cleaned_x_train生成')
    y_train.to_csv('cleaned_data/cleaned_y_train.csv')
    print('cleaned_y_train生成')
    x_test.to_csv('cleaned_data/cleaned_x_test.csv')
    print('cleaned_x_test生成')
    print('数据生成完毕'.center(50, '*'))


def get_result():
    # ---------------------------------------读取数据-----------------------------------------
    x_train = pd.read_csv('cleaned_data/cleaned_x_train.csv')
    x_test = pd.read_csv('cleaned_data/cleaned_x_test.csv')
    y_train = pd.read_csv('cleaned_data/cleaned_y_train.csv')
    blender = Blend(x_train, x_test, y_train)
    blender.blending()
    scores = blender.score()
    print(scores)
    prediction = pd.DataFrame()
    prediction['Sales'] = blender.prediction()
    prediction.to_csv('output/prediction.csv')


func_dic = {
    '1': get_cleaned_data,
    '2': get_result
}
if __name__ == '__main__':
    while True:
        print("""
        1.数据清理&特征工程
        2.获取预测结果
        """)
        choice = input('请输入编号:').strip()
        if choice in func_dic:
            func_dic[choice]()
        elif choice == 'q':
            break
