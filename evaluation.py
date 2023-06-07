import matplotlib.pyplot as plt
import pandas as pd


#FoldData를 이용, fold전체 데이터 셋에 대해 dataframe 생성
def create_dataframe(FoldData):

    dic = {'pred_index':FoldData.pred_index,'y_x':FoldData.y_x,'y_y': FoldData.y_y,
            'pred_x': FoldData.pred_x,'pred_y': FoldData.pred_y,
            'distance': FoldData.distance,'is_correspond': FoldData.is_correspond}

    df = pd.DataFrame(dic) 

    return df

#FoldData를 이용, fold전체 데이터 셋 중 y값과 예측값이 다른 데이터들에 대해 dataframe 생성
def create_falseOnly_dataframe(FoldData):

    dic = {'pred_index':FoldData.pred_index,'y_x':FoldData.y_x,'y_y': FoldData.y_y,
            'pred_x': FoldData.pred_x,'pred_y': FoldData.pred_y,
            'distance': FoldData.distance,'is_correspond': FoldData.is_correspond}

    df = pd.DataFrame(dic) 
    df = df[df.is_correspond != True].sort_values(by='distance' ,ascending=True)

    return df

#falseOnly 데이터의 dataframe을 이용해 산술값 추출
def calculate_falseOnly_statistics(df):
    statistics = {}
    statistics['max'] = df['distance'].max()
    statistics['min'] = df['distance'].min()
    statistics['mode'] = df['distance'].mode()
    statistics['median'] = df['distance'].median()
    statistics['mean'] = df['distance'].mean()
    statistics['var'] = df['distance'].var()
    statistics['std'] = df['distance'].std()
    
    df = pd.DataFrame(statistics) 
    return df

#전체 데이터의 dataframe을 이용해 산술값 추출
def calculate_statistics(df,acc):
    statistics = {}
    statistics['max'] = df['distance'].max()
    statistics['min'] = df['distance'].min()
    statistics['mode'] = df['distance'].mode()
    statistics['median'] = df['distance'].median()
    statistics['mean'] = df['distance'].mean()
    statistics['var'] = df['distance'].var()
    statistics['std'] = df['distance'].std()
    statistics['totalData'] = df.shape[0]
    statistics['falseData'] = df[df.is_correspond != True].shape[0]
    statistics['Accuracy'] = acc

    df = pd.DataFrame(statistics) 
    return df

#거리차 값 이용, 데이터 분포도 그리는 함수
def draw_histogram(dataframe, filepath):
    plt.hist(dataframe['distance'])
    # plt.axis([0, c.DIAGONOL, 0, 3892])
    plt.xlabel('Distance')
    plt.ylabel('Data')
    plt.savefig(filepath)
    plt.close()


#파일로 df 모두 저장
def main(FoldData_train,FoldData_test,modelname,fold,train_acc,test_acc):
    
    traindf = create_dataframe(FoldData_train)
    testdf = create_dataframe(FoldData_test)

    train_falseOnly_df = create_falseOnly_dataframe(FoldData_train)
    test_falseOnly_df = create_falseOnly_dataframe(FoldData_test)

    train_statics = calculate_statistics(traindf,train_acc)
    test_statics = calculate_statistics(testdf,test_acc)

    train_falseOnly_statics = calculate_falseOnly_statistics(train_falseOnly_df)
    test_falseOnly_statics = calculate_falseOnly_statistics(test_falseOnly_df)
    

    # 출력
    train_falseOnly_df.to_csv(f'./{modelname}/fold{fold}/train_data_falseOnly.csv')
    test_falseOnly_df.to_csv(f'./{modelname}/fold{fold}/test_data_falseOnly.csv')

    train_statics.to_csv(f'./{modelname}/fold{fold}/train_statics.csv')
    test_statics.to_csv(f'./{modelname}/fold{fold}/test_statics.csv')

    train_falseOnly_statics.to_csv(f'./{modelname}/fold{fold}/train_statics_falseOnly.csv')
    test_falseOnly_statics.to_csv(f'./{modelname}/fold{fold}/test_statics_falseOnly.csv')

    draw_histogram(train_falseOnly_df,f'./{modelname}/fold{fold}/histogram_train')
    draw_histogram(test_falseOnly_df,f'./{modelname}/fold{fold}/histogram_test')
