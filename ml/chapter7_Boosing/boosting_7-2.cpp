// boosting_letter_rec.cpp : 定义控制台应用程序的入口点。
//
#pragma warning(disable : 4996)
#include "stdafx.h"
#include <opencv.hpp>  
#include <ml/ml.hpp>  
#include <iostream> 
#include <ctime>
using namespace std;
using namespace cv;
using namespace cv::ml;

// 读取数据文件
static bool read_num_class_data(const string& filename, int var_count,
	Mat* _data, Mat* _responses);

int main(int argc, char *argv[])
{

	time_t now = time(0); //获取当前系统的当前日期或时间
	char* dt = ctime(&now); //把now转换为字符串形式
	string data_filename;

	data_filename = argc >= 2 ? argv[1] :
		"../letter-recognition/letter-recognition.data";

	const int class_count = 26;
	Mat data;
	Mat responses;
	Mat weak_responses;

	bool ok = read_num_class_data(data_filename, 16, &data, &responses);
	if (!ok)
		return ok;

	int i, j, k;
	Ptr<Boost> boost;

	int nsamples_all = data.rows; //样本总数:20000
	int ntrain_samples = (int)(nsamples_all*0.5); //一半样本用于训练
	int var_count = data.cols; //特征维数：16

							   //Create or load Boosted Tree classifier
	Mat new_data(ntrain_samples*class_count, var_count + 1, CV_32F);
	//数据扩充26倍
	Mat new_responses(ntrain_samples*class_count, 1, CV_32S);
	//标签对应扩充
	//1. unroll the database type mask
	now = time(0);     //获取当前系统的当前日期或时间
	dt = ctime(&now);    //把 now 转换为字符串形式
	std::cout << dt << " Unrolling the database..." << endl;
	for (i = 0; i < ntrain_samples; i++) //遍历训练集
	{
		const float* data_row = data.ptr<float>(i); //每次指向原始数据的一行
		for (j = 0; j < class_count; j++)  //变量j小于类别数，也就是复制的次数
		{
			float* new_data_row = (float*)new_data.ptr<float>(i*class_count + j);
			//找到新分配的数据的行
			memcpy(new_data_row, data_row, var_count * sizeof(data_row[0]));
			//将每行的特征复制过来
			new_data_row[var_count] = (float)j; //在新数据的末尾增加复制的序号j
			new_responses.at<int>(i*class_count + j) = responses.at<int>(i) == j
				+ 'A';
		}
	}

	Mat var_type(1, var_count + 2, CV_8U);
	//加两维，一维用于原来的标签，一维用于现在的0,1标签
	var_type.setTo(Scalar::all(VAR_ORDERED));  //这些都是数值型的
	var_type.at<uchar>(var_count) = var_type.at<uchar>(var_count + 1) =
		VAR_CATEGORICAL;

	cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(
		new_data, //扩展的26倍向量
		ROW_SAMPLE, //按行存储特征向量，每行为一个特征向量
		new_responses, //扩展的响应
		cv::noArray(), //指定用于训练的变量索引的向量, 此处为全部变量均参与训练
		cv::noArray(), //指定用于训练的样本索引的向量, 此处为全部样本均参与训练
		cv::noArray(), //带有每个样本权重的可选向量, 此处为所有样本权重相同
		var_type //可选的向量, 包含每个输入和输出变量的类型，现在有16+2个条目
	);
	vector<double> priors(2);
	priors[0] = 1;
	priors[1] = 26;
	//经扩展，现有16+2个项目
	now = time(0);
	dt = ctime(&now);
	std::cout << dt << " Training the classifier (may take a few minutes)...\n";
	boost = Boost::create();
	boost->setBoostType(Boost::GENTLE);
	boost->setWeakCount(100);
	boost->setWeightTrimRate(0.95);
	boost->setMaxDepth(5);
	boost->setUseSurrogates(false);
	boost->setPriors(Mat(priors));
	boost->train(tdata);


	Mat temp_sample(1, var_count + 1, CV_32F);
	float* tptr = temp_sample.ptr<float>();
	now = time(0);
	dt = ctime(&now);
	cout << dt << " Testing...\n";

	//compute prediction error on train and test data
	double train_hr = 0, test_hr = 0;
	for (i = 0; i < nsamples_all; i++)
	{
		int best_class = 0;
		double max_sum = -DBL_MAX;
		const float* ptr = data.ptr<float>(i);
		for (k = 0; k < var_count; k++)
			tptr[k] = ptr[k];

		for (j = 0; j < class_count; j++)
		{
			tptr[var_count] = (float)j;
			float s = boost->predict(temp_sample, noArray(),
				StatModel::RAW_OUTPUT);
			if (max_sum < s)
			{
				max_sum = s;
				best_class = j + 'A';
			}
		}

		double r = std::abs(best_class - responses.at<int>(i)) < FLT_EPSILON ?
			1 : 0;
		if (i < ntrain_samples)
			train_hr += r;
		else
			test_hr += r;
	}

	test_hr /= nsamples_all - ntrain_samples;
	train_hr = ntrain_samples > 0 ? train_hr / ntrain_samples : 1.;
	now = time(0);
	dt = ctime(&now);
	cout << dt;
	printf("Recognition rate: train = %.1f%%, test = %.1f%%\n",
		train_hr*100., test_hr*100.);
	system("PAUSE");
	return 0;
}

// This function reads data and responses from the file <filename>
static bool read_num_class_data(const string& filename, int var_count,
								Mat* _data, Mat* _responses)
{
	const int M = 1024;
	char buf[M + 2];

	Mat el_ptr(1, var_count, CV_32F);
	int i;
	vector<int> responses;

	_data->release();
	_responses->release();

	FILE* f = fopen(filename.c_str(), "rt");
	if (!f)
	{
		cout << "Could not read the database " << filename << endl;
		return false;
	}

	for (;;)
	{
		char* ptr;
		if (!fgets(buf, M, f) || !strchr(buf, ','))
			break;
		responses.push_back((int)buf[0]);
		ptr = buf + 2;
		for (i = 0; i < var_count; i++)
		{
			int n = 0;
			sscanf(ptr, "%f%n", &el_ptr.at<float>(i), &n);
			ptr += n + 1;
		}
		if (i < var_count)
			break;
		_data->push_back(el_ptr);
	}
	fclose(f);
	Mat(responses).copyTo(*_responses);

	cout << "The database " << filename << " is loaded.\n";

	return true;
}