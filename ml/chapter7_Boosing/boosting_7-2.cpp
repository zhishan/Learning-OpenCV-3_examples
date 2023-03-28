// boosting_letter_rec.cpp : �������̨Ӧ�ó������ڵ㡣
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

// ��ȡ�����ļ�
static bool read_num_class_data(const string& filename, int var_count,
	Mat* _data, Mat* _responses);

int main(int argc, char *argv[])
{

	time_t now = time(0); //��ȡ��ǰϵͳ�ĵ�ǰ���ڻ�ʱ��
	char* dt = ctime(&now); //��nowת��Ϊ�ַ�����ʽ
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

	int nsamples_all = data.rows; //��������:20000
	int ntrain_samples = (int)(nsamples_all*0.5); //һ����������ѵ��
	int var_count = data.cols; //����ά����16

							   //Create or load Boosted Tree classifier
	Mat new_data(ntrain_samples*class_count, var_count + 1, CV_32F);
	//��������26��
	Mat new_responses(ntrain_samples*class_count, 1, CV_32S);
	//��ǩ��Ӧ����
	//1. unroll the database type mask
	now = time(0);     //��ȡ��ǰϵͳ�ĵ�ǰ���ڻ�ʱ��
	dt = ctime(&now);    //�� now ת��Ϊ�ַ�����ʽ
	std::cout << dt << " Unrolling the database..." << endl;
	for (i = 0; i < ntrain_samples; i++) //����ѵ����
	{
		const float* data_row = data.ptr<float>(i); //ÿ��ָ��ԭʼ���ݵ�һ��
		for (j = 0; j < class_count; j++)  //����jС���������Ҳ���Ǹ��ƵĴ���
		{
			float* new_data_row = (float*)new_data.ptr<float>(i*class_count + j);
			//�ҵ��·�������ݵ���
			memcpy(new_data_row, data_row, var_count * sizeof(data_row[0]));
			//��ÿ�е��������ƹ���
			new_data_row[var_count] = (float)j; //�������ݵ�ĩβ���Ӹ��Ƶ����j
			new_responses.at<int>(i*class_count + j) = responses.at<int>(i) == j
				+ 'A';
		}
	}

	Mat var_type(1, var_count + 2, CV_8U);
	//����ά��һά����ԭ���ı�ǩ��һά�������ڵ�0,1��ǩ
	var_type.setTo(Scalar::all(VAR_ORDERED));  //��Щ������ֵ�͵�
	var_type.at<uchar>(var_count) = var_type.at<uchar>(var_count + 1) =
		VAR_CATEGORICAL;

	cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(
		new_data, //��չ��26������
		ROW_SAMPLE, //���д洢����������ÿ��Ϊһ����������
		new_responses, //��չ����Ӧ
		cv::noArray(), //ָ������ѵ���ı�������������, �˴�Ϊȫ������������ѵ��
		cv::noArray(), //ָ������ѵ������������������, �˴�Ϊȫ������������ѵ��
		cv::noArray(), //����ÿ������Ȩ�صĿ�ѡ����, �˴�Ϊ��������Ȩ����ͬ
		var_type //��ѡ������, ����ÿ�������������������ͣ�������16+2����Ŀ
	);
	vector<double> priors(2);
	priors[0] = 1;
	priors[1] = 26;
	//����չ������16+2����Ŀ
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