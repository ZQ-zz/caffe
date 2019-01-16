#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/stage2_multibox_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Stage2_MultiBoxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
    this->layer_param_.add_propagate_down(false);
  }
  const MultiBoxLossParameter& multibox_loss_param =
      this->layer_param_.multibox_loss_param();
  multibox_loss_param_ = this->layer_param_.multibox_loss_param();

  num_ = bottom[0]->num();
  num_priors_ = bottom[2]->height() / 4;
  // Get other parameters.
  CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  num_classes_ = multibox_loss_param.num_classes();
  CHECK_GE(num_classes_, 1) << "num_classes should not be less than 1.";
  share_location_ = multibox_loss_param.share_location();
  loc_classes_ = share_location_ ? 1 : num_classes_;
  background_label_id_ = multibox_loss_param.background_label_id();
  use_difficult_gt_ = multibox_loss_param.use_difficult_gt();
  mining_type_ = multibox_loss_param.mining_type();
  if (multibox_loss_param.has_do_neg_mining()) {
    LOG(WARNING) << "do_neg_mining is deprecated, use mining_type instead.";
    do_neg_mining_ = multibox_loss_param.do_neg_mining();
    CHECK_EQ(do_neg_mining_,
             mining_type_ != MultiBoxLossParameter_MiningType_NONE);
  }
  do_neg_mining_ = mining_type_ != MultiBoxLossParameter_MiningType_NONE;

  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

  if (do_neg_mining_) {
    CHECK(share_location_)
        << "Currently only support negative mining if share_location is true.";
  }

  vector<int> loss_shape(1, 1);
  // Set up localization loss layer.
  loc_weight_ = multibox_loss_param.loc_weight();
  loc_loss_type_ = multibox_loss_param.loc_loss_type();
  // fake shape.
  vector<int> loc_shape(1, 1);
  loc_shape.push_back(4);
  loc_pred_.Reshape(loc_shape);
  loc_gt_.Reshape(loc_shape);
  loc_bottom_vec_.push_back(&loc_pred_);
  loc_bottom_vec_.push_back(&loc_gt_);
  loc_loss_.Reshape(loss_shape);
  loc_top_vec_.push_back(&loc_loss_);
  if (loc_loss_type_ == MultiBoxLossParameter_LocLossType_L2) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_l2_loc");
    layer_param.set_type("EuclideanLoss");
    layer_param.add_loss_weight(loc_weight_);
    loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
  } else if (loc_loss_type_ == MultiBoxLossParameter_LocLossType_SMOOTH_L1) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_smooth_L1_loc");
    layer_param.set_type("SmoothL1Loss");
    layer_param.add_loss_weight(loc_weight_);
    loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
  } else {
    LOG(FATAL) << "Unknown localization loss type.";
  }
  // Set up confidence loss layer.
  conf_loss_type_ = multibox_loss_param.conf_loss_type();
  conf_bottom_vec_.push_back(&conf_pred_);
  conf_bottom_vec_.push_back(&conf_gt_);
  conf_loss_.Reshape(loss_shape);
  conf_top_vec_.push_back(&conf_loss_);
  if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    CHECK_GE(background_label_id_, 0)
        << "background_label_id should be within [0, num_classes) for Softmax.";
    CHECK_LT(background_label_id_, num_classes_)
        << "background_label_id should be within [0, num_classes) for Softmax.";
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_conf");
    layer_param.set_type("SoftmaxWithLoss");
    layer_param.add_loss_weight(Dtype(1.));
    layer_param.mutable_loss_param()->set_normalization(
        LossParameter_NormalizationMode_NONE);
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(1);
    // Fake reshape.
    vector<int> conf_shape(1, 1);
    conf_gt_.Reshape(conf_shape);
    conf_shape.push_back(num_classes_);
    conf_pred_.Reshape(conf_shape);
    conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
  } else if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_logistic_conf");
    layer_param.set_type("SigmoidCrossEntropyLoss");
    layer_param.add_loss_weight(Dtype(1.));
    // Fake reshape.
    vector<int> conf_shape(1, 1);
    conf_shape.push_back(num_classes_);
    conf_gt_.Reshape(conf_shape);
    conf_pred_.Reshape(conf_shape);
    conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
}

template <typename Dtype>
void Stage2_MultiBoxLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  num_ = bottom[4]->num();
  num_priors_ = bottom[2]->height() / 4;
  num_gt_ = bottom[3]->height();
//  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
//  CHECK_EQ(num_priors_ * loc_classes_ * 4, bottom[0]->channels())
//      << "Number of priors must match number of location predictions.";
//  CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
//      << "Number of priors must match number of confidence predictions.";
}

template <typename Dtype>
void Stage2_MultiBoxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
//  const Dtype* loc_data = bottom[0]->cpu_data();
//  const Dtype* conf_data = bottom[1]->cpu_data();
//  const Dtype* prior_data = bottom[2]->cpu_data();
  const Dtype* gt_data = bottom[3]->cpu_data(); //

  // Retrieve all ground truth.
  map<int, vector<NormalizedBBox> > all_gt_bboxes;
  GetGroundTruth(gt_data, num_gt_, background_label_id_, use_difficult_gt_,
                 &all_gt_bboxes);

#if 0
  // Retrieve all prior bboxes. It is same within a batch since we assume all
  // images in a batch are of same dimension.
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

  // Retrieve all predictions.
  vector<LabelBBox> all_loc_preds;
  GetLocPredictions(loc_data, num_, num_priors_, loc_classes_, share_location_,
                    &all_loc_preds);

  // Find matches between source bboxes and ground truth bboxes.
  vector<map<int, vector<float> > > all_match_overlaps;
  FindMatches(all_loc_preds, all_gt_bboxes, prior_bboxes, prior_variances,
              multibox_loss_param_, &all_match_overlaps, &all_match_indices_);

  num_matches_ = 0;
  int num_negs = 0;
  // Sample hard negative (and positive) examples based on mining type.
  MineHardExamples(*bottom[1], all_loc_preds, all_gt_bboxes, prior_bboxes,
                   prior_variances, all_match_overlaps, multibox_loss_param_,
                   &num_matches_, &num_negs, &all_match_indices_,
                   &all_neg_indices_);

#endif

  ///////////////////////////////////////////
  // show debug
#if 0
	Blob<Dtype> &blob_img = *bottom[4];
    int num = num_;
	char text[260];
	for (int n = 0; n < num; n++)
	{
		int channel = blob_img.channels();
		int height = blob_img.height();
		int width = blob_img.width();
		cv::Mat img(height, width, CV_8UC3);

		// get net input image
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				for (int k = 0; k < channel; k++)
				{
					//LOG(INFO) << "test: " << i << j << k;
					img.at<cv::Vec3b>(i, j)[k] = (float)blob_img.data_at(n, k, i, j);
				}
			}
		}

		// draw ground truth bbox
		vector<NormalizedBBox> gt_bboxes = all_gt_bboxes[n];
		for (vector<NormalizedBBox>::iterator iter = gt_bboxes.begin(); iter != gt_bboxes.end(); iter++)
		{
			cv::Point pt1, pt2;
			CvScalar c = CV_RGB(255, 0, 0); // red
			pt1.x = round(iter->xmin()* img.cols);
			pt1.y = round(iter->ymin() * img.rows);
			pt2.x = round(iter->xmax() * img.cols);
			pt2.y = round(iter->ymax() * img.rows);
			cv::rectangle(img, pt1, pt2, c, 1, 8, 0);
		}

		// draw stage1 detection out
//		vector<NormalizedBBox> gt_bboxes = all_gt_bboxes[n];
		const Dtype* det_data = bottom[2]->cpu_data();
		int idx = 0;
		for (int i = 0; i < bottom[2]->height(); i++)
		{
			if (n == det_data[0])
			{
				idx++;
				cv::Point pt1, pt2;
				CvScalar c = CV_RGB(0, 255, 0); // green
				pt1.x = round(det_data[3] * img.cols);
				pt1.y = round(det_data[4] * img.rows);
				pt2.x = round(det_data[5] * img.cols);
				pt2.y = round(det_data[6] * img.rows);
				cv::rectangle(img, pt1, pt2, c, 1, 8, 0);
			}
			det_data += 7;
		}
		LOG(INFO) << "idx = " << idx << std::endl;
		LOG(INFO) << "bottom[0]->shape_string() = " << bottom[0]->shape_string() << std::endl;
		LOG(INFO) << "bottom[1]->shape_string() = " << bottom[1]->shape_string() << std::endl;
		LOG(INFO) << "bottom[2]->shape_string() = " << bottom[2]->shape_string() << std::endl;

		int key;
		cv::imshow("DetectRst", img);
		key = cv::waitKey(0);

	}
#endif

	//
//	  bottom: "stage2_loc/flat"
//	  bottom: "stage2_conf/flat"
//	  bottom: "detection_out"
//	  bottom: "detection_out_enlarge"
//	  bottom: "label"
//	  bottom: "data"
//	  top: "stage2_mbox_loss"
	//loc loss and conf loss
	{
//		LOG(INFO) << "bottom[0]->shape_string() = " << bottom[0]->shape_string() << std::endl;
//		LOG(INFO) << "bottom[1]->shape_string() = " << bottom[1]->shape_string() << std::endl;
//		LOG(INFO) << "bottom[2]->shape_string() = " << bottom[2]->shape_string() << std::endl;

	    Dtype* loc_pred_data = loc_pred_.mutable_cpu_data();
	    Dtype* loc_gt_data = loc_gt_.mutable_cpu_data();
	    Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();
	    Dtype* conf_gt_data = conf_gt_.mutable_cpu_data();
		//int image_batch = num_;
		// encode loc
		int num_loc = bottom[0]->count() / 4;
		CHECK_EQ(num_loc, bottom[2]->shape(2));
//		CHECK_EQ(num_loc, bottom[3]->shape(2));
//		const Dtype *det_enlarge = bottom[3]->cpu_data();
		const Dtype *det = bottom[2]->cpu_data();
		const Dtype *conf = bottom[1]->cpu_data();
		const Dtype *loc = bottom[0]->cpu_data();
		match_indices_.clear();
		int idx = 0;
		for (int i = 0; i < num_loc; i++)
		{
			// get cur loc
			int stage1_batch = det[0];
			NormalizedBBox stage1_bbox;
			stage1_bbox.set_label(det[1]);
			stage1_bbox.set_xmin(det[3]);
			stage1_bbox.set_ymin(det[4]);
			stage1_bbox.set_xmax(det[5]);
			stage1_bbox.set_ymax(det[6]);
			int maxid = -1;
			int label = 0;
			float maxVal = 0.0f;
			float overlap = 0.0f;
			// get max overlap and gt id
//			CHECK_LT(stage1_batch, all_gt_bboxes.size());
			if (all_gt_bboxes.count(stage1_batch) > 0)
			{
				for (int j = 0; j < all_gt_bboxes[stage1_batch].size(); j++)
				{
					NormalizedBBox cur_gt_bbox = all_gt_bboxes[stage1_batch][j];
					overlap = JaccardOverlap(stage1_bbox, cur_gt_bbox, true);
					if (overlap > maxVal)
					{
						maxVal = overlap;
						maxid = j;
						label = cur_gt_bbox.label();
					}
				}
			}

#if 0
			// get image
			Blob<Dtype> &blob_img = *bottom[4];
			int channel = blob_img.channels();
			int height = blob_img.height();
			int width = blob_img.width();
			cv::Mat img(height, width, CV_8UC3);

			// get net input image
			{
				for (int ii = 0; ii < height; ii++)
				{
					for (int jj = 0; jj < width; jj++)
					{
						for (int kk = 0; kk < channel; kk++)
						{
							//LOG(INFO) << "test: " << i << j << k;
							img.at<cv::Vec3b>(ii, jj)[kk] = (float)blob_img.data_at(stage1_batch, kk, ii, jj);
						}
					}
				}
			}
#endif

			if (maxVal > 0.5f)
			{
				//encode
			    float prior_width = det[5] - det[3];
			    CHECK_GT(prior_width, 0);
			    float prior_height = det[6] - det[4];
			    CHECK_GT(prior_height, 0);
			    float prior_center_x = (det[5] + det[3]) / 2.;
			    float prior_center_y = (det[6] + det[4]) / 2.;

			    float gt_xmin = all_gt_bboxes[stage1_batch][maxid].xmin();
			    float gt_ymin = all_gt_bboxes[stage1_batch][maxid].ymin();
			    float gt_xmax = all_gt_bboxes[stage1_batch][maxid].xmax();
			    float gt_ymax = all_gt_bboxes[stage1_batch][maxid].ymax();

			    float bbox_width = gt_xmax - gt_xmin;
			    CHECK_GT(bbox_width, 0);
			    float bbox_height = gt_ymax - gt_ymin;
			    CHECK_GT(bbox_height, 0);
			    float bbox_center_x = (gt_xmin + gt_xmax) / 2.;
			    float bbox_center_y = (gt_ymin + gt_ymax) / 2.;

			    loc_pred_data[idx*4 + 0] = loc[0];
			    loc_pred_data[idx*4 + 1] = loc[1];
			    loc_pred_data[idx*4 + 2] = loc[2];
			    loc_pred_data[idx*4 + 3] = loc[3];

				loc_gt_data[idx*4 + 0] = (bbox_center_x - prior_center_x) / prior_width / 0.1;
				loc_gt_data[idx*4 + 1] = (bbox_center_y - prior_center_y) / prior_height / 0.1;
				loc_gt_data[idx*4 + 2] = log(bbox_width / prior_width) / 0.2;
				loc_gt_data[idx*4 + 3] = log(bbox_height / prior_height) / 0.2;

#if 0
				LOG(INFO) << "loc_pred_data:";
				for (int k = 0; k < 4; k++)
				{
				    LOG(INFO) << loc_pred_data[idx*4 + k] << " ";
				}
				LOG(INFO) << std::endl;
				LOG(INFO) << "loc_gt_data:";
				for (int k = 0; k < 4; k++)
				{
				    LOG(INFO) << loc_gt_data[idx*4 + k] << " ";
				}
				LOG(INFO) << std::endl;
#endif

#if 0
				cv::Point pt1, pt2;
				CvScalar c;
				c = CV_RGB(0, 255, 0); // green
				pt1.x = round(det[3] * img.cols);
				pt1.y = round(det[4] * img.rows);
				pt2.x = round(det[5] * img.cols);
				pt2.y = round(det[6] * img.rows);
				cv::rectangle(img, pt1, pt2, c, 1, 8, 0);

				c = CV_RGB(255, 0, 0); // green
				pt1.x = round(gt_xmin * img.cols);
				pt1.y = round(gt_ymin * img.rows);
				pt2.x = round(gt_xmax * img.cols);
				pt2.y = round(gt_ymax * img.rows);
				cv::rectangle(img, pt1, pt2, c, 1, 8, 0);

				int key;
				cv::imshow("debug", img);
				//key = cv::waitKey(0);
#endif

				match_indices_.push_back(i); // for backward

				idx++;
			}
			else
			{

			}

			// conf
			conf_pred_data[i*4 + 0] = conf[0];
			conf_pred_data[i*4 + 1] = conf[1];
			conf_pred_data[i*4 + 2] = conf[2];
			conf_pred_data[i*4 + 3] = conf[3];

			conf_gt_data[i] = label;

#if 0
			LOG(INFO) << "conf_pred_data:";
			for (int k = 0; k < 4; k++)
			{
				LOG(INFO) << conf_pred_data[i*4 + k] << " ";
			}
			LOG(INFO) << std::endl;
			LOG(INFO) << "conf_gt_data:" << conf_gt_data[i] << std::endl;
			LOG(INFO) << "stage1 det:" << det[1] << std::endl;

#endif

			//update
//			det_enlarge += 7;
			det += 7;
			loc += 4;
			conf += 4;

		}

		// loc
		num_matches_ = idx;
		if (num_matches_ > 0)
		{
			vector<int> loc_shape(2);
			loc_shape[0] = 1;
			loc_shape[1] = num_matches_ * 4; //
			loc_pred_.Reshape(loc_shape);
			loc_gt_.Reshape(loc_shape);
			loc_loss_layer_->Reshape(loc_bottom_vec_, loc_top_vec_);
			loc_loss_layer_->Forward(loc_bottom_vec_, loc_top_vec_);
		}
		else
		{
			loc_loss_.mutable_cpu_data()[0] = 0;
		}

		// conf
		num_conf_ = num_loc;
		if (num_conf_ > 0)
		{
			vector<int> conf_shape;
			conf_shape.push_back(num_loc);
			conf_gt_.Reshape(conf_shape);
			conf_shape.push_back(4); //num_classes_
			conf_pred_.Reshape(conf_shape);
			conf_loss_layer_->Reshape(conf_bottom_vec_, conf_top_vec_);
			conf_loss_layer_->Forward(conf_bottom_vec_, conf_top_vec_);
		}
		else
		{
			conf_loss_.mutable_cpu_data()[0] = 0;
		}

		// normalize
	    top[0]->mutable_cpu_data()[0] = 0;
	    CHECK_EQ(this->layer_param_.propagate_down(0), true);
	    if (this->layer_param_.propagate_down(0))
	    {
			Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
			  normalization_, num_, num_priors_, num_matches_);
			top[0]->mutable_cpu_data()[0] +=
			  loc_weight_ * loc_loss_.cpu_data()[0] / normalizer;

//			LOG(INFO) << "normalizer: " << normalizer;
//			LOG(INFO) << "loc_weight_: " << loc_weight_;
//			LOG(INFO) << "loc_loss_: " << loc_loss_.cpu_data()[0];
//			LOG(INFO) << "stage2 loc loss: " << loc_loss_.cpu_data()[0] / normalizer;

//			for (int ii = 0; ii < num_matches_ * 4; ii++)
//			{
//				LOG(INFO) << loc_pred_data[ii] << " " << loc_gt_data[ii];
//			}
	    }


	    CHECK_EQ(this->layer_param_.propagate_down(1), true);
	    if (this->layer_param_.propagate_down(1))
	    {
			Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
			  normalization_, num_, num_priors_, num_conf_);
			top[0]->mutable_cpu_data()[0] += conf_loss_.cpu_data()[0] / normalizer;
//		    LOG(INFO) << "normalizer: " << normalizer;
//		    LOG(INFO) << "conf_loss_: " << conf_loss_.cpu_data()[0];
//		    LOG(INFO) << "stage2 conf loss: " << conf_loss_.cpu_data()[0] / normalizer;
	    }

#if 0
		//
		cv::waitKey(0);
#endif

	}


#if 0
  ShowPosNegBBoxes(*bottom[4], num_, prior_bboxes, all_gt_bboxes,
		  all_match_indices_, all_neg_indices_);
#endif
  ///////////////////////////////////////////

}

template <typename Dtype>
void Stage2_MultiBoxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to prior inputs.";
  }
  if (propagate_down[3]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }

  //loc
  CHECK_EQ(propagate_down[0], true);
  CHECK_EQ(4 * num_conf_, bottom[0]->count());
  if (propagate_down[0])
  {
	  Dtype* loc_bottom_diff = bottom[0]->mutable_cpu_diff();
	  caffe_set(bottom[0]->count(), Dtype(0), loc_bottom_diff);
	  CHECK_EQ(num_matches_, match_indices_.size());
	  if (num_matches_ > 0)
	  {
	      vector<bool> loc_propagate_down;
	      // Only back propagate on prediction, not ground truth.
	      loc_propagate_down.push_back(true);
	      loc_propagate_down.push_back(false);
	      loc_loss_layer_->Backward(loc_top_vec_, loc_propagate_down,
	                                loc_bottom_vec_);

	      // Scale gradient.
	      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
	          normalization_, num_, num_priors_, num_matches_);
	      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
	      caffe_scal(loc_pred_.count(), loss_weight, loc_pred_.mutable_cpu_diff());

	      // Copy gradient back to bottom[0].
	      const Dtype* loc_pred_diff = loc_pred_.cpu_diff();
	      for (int i = 0; i < num_matches_; i++)
	      {
	    	  int index = match_indices_[i];
	          caffe_copy<Dtype>(4, loc_pred_diff + i*4, loc_bottom_diff + index*4);
	      }
	  }

  }

  // conf
  CHECK_EQ(propagate_down[1], true);
  CHECK_EQ(num_classes_, 4);
  CHECK_EQ(num_classes_*num_conf_, bottom[1]->count());
  if (propagate_down[1])
  {
	  Dtype* conf_bottom_diff = bottom[1]->mutable_cpu_diff();
	  caffe_set(bottom[1]->count(), Dtype(0), conf_bottom_diff);

	  if (num_conf_ > 0)
	  {
		  vector<bool> conf_propagate_down;
	      // Only back propagate on prediction, not ground truth.
	      conf_propagate_down.push_back(true);
	      conf_propagate_down.push_back(false);
	      conf_loss_layer_->Backward(conf_top_vec_, conf_propagate_down,
	                                 conf_bottom_vec_);

	      // Scale gradient.
	      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
	          normalization_, num_, num_priors_, num_conf_);
	      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
	      caffe_scal(conf_pred_.count(), loss_weight,
	                 conf_pred_.mutable_cpu_diff());

	      // Copy gradient back to bottom[1].
	      const Dtype* conf_pred_diff = conf_pred_.cpu_diff();

	      for (int i = 0; i < num_conf_; i++)
	      {
              // Copy the diff to the right place.
              caffe_copy<Dtype>(num_classes_, conf_pred_diff + i * num_classes_,
            		  conf_bottom_diff + i * num_classes_);
	      }
	  }

  }

  // After backward, remove match statistics.
  all_match_indices_.clear();
  all_neg_indices_.clear();
}

INSTANTIATE_CLASS(Stage2_MultiBoxLossLayer);
REGISTER_LAYER_CLASS(Stage2_MultiBoxLoss);

}  // namespace caffe
