// Creating and training a decision tree

#include <opencv2/opencv.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <iostream> 

using namespace std;
using namespace cv;
using namespace cv::ml;

int main(int argc, char *argv[]) {
	// 1.读取数据
	// 1.1 读取训练集
	const char *csv_file_name_train = "../Datasets/boston-house-prices/housing-train.csv";
	cv::Ptr<TrainData> dataSetTrain =
		TrainData::loadFromCSV(csv_file_name_train, // Input file name
			0, // 从数据文件开头跳过的行数
			-1, // 样本标签于此列开始，-1时样本标签为最后一列
			-1 //  样本标签于此列结束，-1时为上一个参数所在列	
		);
	// 验证数据读取是否正确
	int n_train_samples = dataSetTrain->getNSamples();
	if (n_train_samples == 0) {
		cerr << "读取文件错误: " << csv_file_name_train << endl;
		exit(-1);
	}
	else {
		cout << "从" << csv_file_name_train << "中，读取了" << n_train_samples << "个训练样本" << endl;
	}
	// 1.2 读取测试集
	const char *csv_file_name_test = "../Datasets/boston-house-prices/housing-test.csv";
	cv::Ptr<TrainData> dataSetTest = TrainData::loadFromCSV(csv_file_name_test,	0, -1, -1);
	int n_test_samples = dataSetTest->getNSamples();
	if (n_test_samples == 0) {
		cerr << "读取文件错误: " << csv_file_name_test << endl;
		exit(-1);
	}
	else {
		cout << "从" << csv_file_name_test << "中，读取了" << n_test_samples << "个测试样本" << endl;
	}

	// 2.创建决策树模型
	cv::Ptr<RTrees> dtree = RTrees::create();

	// 3.设置模型参数
	dtree->setMaxDepth(15);//15
	dtree->setMinSampleCount(2);//2
	dtree->setRegressionAccuracy(0.01f);
	dtree->setUseSurrogates(false /* true */);
	dtree->setCalculateVarImportance(true); // 开启特征重要性计算
	//dtree->setMaxCategories(15);
	dtree->setCVFolds(0 /*10*/); // 
	dtree->setUse1SERule(true/*true*/);
	dtree->setTruncatePrunedTree(true);

	// 4.训练决策树
	cout << "start training..." << endl;
	dtree->train(dataSetTrain);
	cout << "training success!" << endl;

	// 输出样本特征属性重要性
	Mat var_importance = dtree->getVarImportance();
	if (!var_importance.empty())
	{
		double rt_imp_sum = sum(var_importance)[0];
		printf("var#\timportance (%%):\n");
		int i, n = (int)var_importance.total();// 返回矩阵的元素总个数
		for (i = 0; i < n; i++)
			printf("%-2d\t%-4.1f\n", i, 100.f*var_importance.at<float>(i) / rt_imp_sum);
	}

	// 5. 训练集精度
	cv::Mat train_results;
	float MSE_train = dtree->calcError(dataSetTrain,
		false, // use train data
		train_results);
	cv::Mat expected_responses = dataSetTrain->getResponses();
	int total_train = 0;
	float square_error = 0.0;
	cout << endl << "--- train set --- " << endl;
	for (int i = 0; i < expected_responses.rows; ++i) {
		float responses = train_results.at<float>(i, 0);
		float expected = expected_responses.at<float>(i, 0);
		square_error += (expected - responses) * (expected - responses);
		total_train++;
		//cout << "price: " << expected << ",\tpredict: " << responses << endl;
		cout << expected << "\t" << responses << endl;
	}
	// 计算RMSE指标
	float RMSE_train = sqrt(square_error / total_train);

	// 6. 保存模型
	dtree->save("trained_dtree.xml");

	// 7. 读取模型
	dtree->load("trained_dtree.xml");

	// 8. 测试集精度
	cv::Mat results_test;
	float MSE_test = dtree->calcError(dataSetTest,
		true, // use train data: now it is test data actually
		results_test);
	cv::Mat expected_responses_test = dataSetTest->getResponses();
	//cout << expected_responses_test.size() << endl;
	int total_test = 0;
	square_error = 0.0;
	cout << endl << "--- test set --- " << endl;
	for (int i = 0; i < expected_responses_test.rows; ++i) {
		float responses = results_test.at<float>(i, 0);
		float expected = expected_responses_test.at<float>(i, 0);
		square_error += (expected - responses) * (expected - responses);
		total_test++;
		//cout << "price: " << expected << ",\tpredict: " << responses << endl;
		cout <<  expected << "\t" << responses << endl;
	}
	// 计算RMSE指标
	float RMSE_test = sqrt(square_error / total_test);
	cout << "train data RMSE  = " << RMSE_train << " k USD" << endl;
	cout << "test  data RMSE  = " << RMSE_test  << " k USD" << endl;
	cout << "train data MSE   = " << MSE_train << endl;
	cout << "test  data MSE   = " << MSE_test << endl;
	return 0;
}
