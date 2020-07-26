/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2015-2016, Jiri Horner.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Jiri Horner nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *********************************************************************/

#include <combine_grids/grid_compositor.h>
#include <combine_grids/grid_warper.h>
#include <combine_grids/merging_pipeline.h>
#include <ros/assert.h>
#include <ros/console.h>
#include "estimation_internal.h"

namespace combine_grids
{
bool MergingPipeline::estimateTransforms(FeatureType feature_type, double confidence, std::vector<std::string> robots, std::string reference_robot)
{
  std::vector<cv::detail::ImageFeatures> image_features;
  std::vector<cv::detail::CameraParams> transforms;
  std::vector<int> good_indices;
  int ref_id = -1;

  // TODO investigate value translation effect on features
  cv::Ptr<cv::detail::FeaturesFinder> finder = internal::chooseFeatureFinder(feature_type);
  cv::Ptr<cv::detail::FeaturesMatcher> matcher = cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>();
  cv::Ptr<cv::detail::Estimator> estimator = cv::makePtr<cv::detail::AffineBasedEstimator>();
  //cv::Ptr<cv::detail::BundleAdjusterBase> adjuster = cv::makePtr<cv::detail::BundleAdjusterAffinePartial>();

  if (images_.empty()) {
    return true;
  }

  /* find features in images */
  ROS_DEBUG("computing features");
  image_features.reserve(images_.size());
  for (const cv::Mat& image : images_) {
    image_features.emplace_back();
    if (!image.empty()) {
      (*finder)(image, image_features.back());
    }
  }
  finder->collectGarbage();

  /* find corespondent features */
  ROS_DEBUG("pairwise matching features");
  (*matcher)(image_features, pairwise_matches_);
  matcher->collectGarbage();

#ifndef NDEBUG
  //internal::writeDebugMatchingInfo(images_, image_features, pairwise_matches_);
#endif

  /* use only matches that has enough confidence. leave out matches that are not
   * connected (small components) */
  good_indices = cv::detail::leaveBiggestComponent(
      image_features, pairwise_matches_, static_cast<float>(confidence));

  /* estimate transform */
  ROS_DEBUG("calculating transforms in global reference frame");
  // note: currently used estimator never fails
  if (!(*estimator)(image_features, pairwise_matches_, transforms)) {
    return false;
  }

  /* levmarq optimization */
  // openCV just accepts float transforms
  for (auto& transform : transforms) {
    transform.R.convertTo(transform.R, CV_32F);
  }

/*
 * The optimization fails horribly if the maps are not from simulations (if maps are noisy)
 *
  ROS_DEBUG("optimizing global transforms");
  adjuster->setConfThresh(confidence);
  if (!(*adjuster)(image_features, pairwise_matches, transforms)) {
    ROS_WARN("Bundle adjusting failed. Could not estimate transforms.");
    return false;
  }
*/

  transforms_.clear();
  transforms_.resize(images_.size());
  size_t i = 0;
  for (auto& j : good_indices) {
    // we want to work with transforms as doubles
    transforms[i].R.convertTo(transforms_[static_cast<size_t>(j)], CV_64FC1);
    ++i;
  }

  /*
   *  Check which robot is chosen as reference (transform there is identity) from merger. Is it the same as the wanted?
   *  If not, change transforms...
   */
  for (size_t i = 0; i < robots.size(); ++i) {
      if(robots.at(i) == reference_robot) {
          ref_id = i;
      }
  }
  if(ref_id < 0) {
      ROS_WARN("Referenced robot seems not to be in robot list.");
      return false;
  }

  // cv wants doubles for matrix inverting
  for(auto tr : transforms_) {
      cv::Mat mat = cv::Mat::eye(3, 3, CV_64F);
      mat = tr.clone();
      mat.assignTo(tr, CV_64F);
  }
  ROS_DEBUG("Try setting reference frame");

  // Check if the reference robot has its transformation yet. Otherwise the tf subscriber fails.

  if (transforms_.at(ref_id).empty()) {
      ROS_INFO("no transform for reference frame available yet");
      return false;
  }

  /*
   * If the reference is not the correct one, the transforms must be calculated to the correct one
   */
  for (size_t i = 0; i < transforms_.size(); ++i) {
      if(transforms_.at(i).empty()) continue;
      cv::Mat diff = transforms_.at(i) != cv::Mat::eye(3, 3, CV_64F);
      if(cv::countNonZero(diff) == 0) {
          if(i != ref_id) {
              cv::Mat Tref_inv = transforms_.at(ref_id).inv();
              for(size_t j = 0; j < transforms_.size(); ++j) {
                  if (transforms_.at(j).empty()) continue;
                  if(j == ref_id) transforms_.at(j) = cv::Mat::eye(3, 3, CV_64F);
                  else transforms_.at(j) = transforms_.at(j) * Tref_inv;
              }
          }
          break;
      }
  }
  ROS_DEBUG("Reference frame set");
  return true;
}


nav_msgs::OccupancyGrid::Ptr MergingPipeline::composeGrids()
{
  if(images_.size() != transforms_.size()) {
    return nullptr;
  }
  if(images_.size() != grids_.size()) {
    return nullptr;
  }
  if (images_.empty()) {
    return nullptr;
  }

  ROS_DEBUG("warping grids");
  internal::GridWarper warper;
  std::vector<cv::Mat> imgs_warped;
  imgs_warped.reserve(images_.size());
  rois_.clear();
  rois_.reserve(images_.size());

  for (size_t i = 0; i < images_.size(); ++i) {
    if (!transforms_[i].empty() && !images_[i].empty()) {
      imgs_warped.emplace_back();
      rois_.emplace_back(warper.warp(images_[i], transforms_[i], imgs_warped.back()));
    }
  }

  if (imgs_warped.empty()) {
    return nullptr;
  }

  ROS_DEBUG("compositing result grid");
  nav_msgs::OccupancyGrid::Ptr result;
  internal::GridCompositor compositor;
  result = compositor.compose(imgs_warped, rois_);
  result->info.map_load_time = ros::Time::now();
  result->info.origin.orientation.w = 1.;
  // TODO is this correct?
  result->info.resolution = grids_[0]->info.resolution;

  return result;
}


std::vector<geometry_msgs::TransformStamped> MergingPipeline::transformsForPublish(std::vector<std::string> robots, double resolution)
{

  std::vector<geometry_msgs::TransformStamped> result;
  cv::Vec2d origin_offset;
  cv::Mat origin_offset_rot;
  cv::Rect r(0, 0, 2, 2);
  result.reserve(transforms_.size());

  ROS_DEBUG("Transform offset");

  /*
   * The size of the merged map is determined in the warping process.
   * To get the offset applied by this process, one need the minimal
   * offset of the roi to warp (as seen in grid_warper.cpp.
   */
  cv::Point2d offset;
  offset.x = 0, offset.y = 0;
  for(auto& r : rois_) {
      offset.x = (r.tl().x < offset.x ? r.tl().x : offset.x);
      offset.y = (r.tl().y < offset.y ? r.tl().y : offset.y);
  }

  for (size_t i = 0; i < transforms_.size(); ++i) {
    if (transforms_.at(i).empty()) continue;

    cv::Mat T_inv, R;
    T_inv = transforms_.at(i).inv();
    R = T_inv(r).clone();

    ROS_ASSERT(transforms_[i].type() == CV_64FC1);
    geometry_msgs::TransformStamped ros_transform;
    ros_transform.header.frame_id = "/world";                   // Transform from ...
    ros_transform.child_frame_id = robots.at(i) + "/map";       // Transform to   ...

    ros_transform.header.stamp = ros::Time::now();

    /*
     * The origin offset provided by the warper is in the map frame.
     * One need to transform it into the world frame to get the correct offsets
     */

    origin_offset[0] = (double)grids_[i]->info.origin.position.x;
    origin_offset[1] = (double)grids_[i]->info.origin.position.y;

    origin_offset_rot = R * cv::Mat(origin_offset);

    ROS_DEBUG("Make Transforms ready for publish");
    ros_transform.transform.translation.x = (T_inv.at<double>(0, 2) - offset.x) * resolution - origin_offset_rot.at<double>(0,0);
    ros_transform.transform.translation.y = (T_inv.at<double>(1, 2) - offset.y) * resolution - origin_offset_rot.at<double>(1,0);
    ros_transform.transform.translation.z = 0.;

    /* To avoid errors caused by scaling (maps are metric, so scaling is always a mistake, caused by to many DOF of affine transformator.
     * We only need 3 DOFs, OpenCVs has 6.
     */
    double a = T_inv.at<double>(0, 0);
    double b = T_inv.at<double>(1, 0);
    double scale;

    scale = sqrt(pow(a, 2) + pow(b, 2));
    if(scale < -1 || scale > 1) {
        a /= scale;
        b /= scale;
    }

    ros_transform.transform.rotation.w = std::sqrt(2. + 2. * a) * 0.5;
    ros_transform.transform.rotation.x = 0.;
    ros_transform.transform.rotation.y = 0.;
    ros_transform.transform.rotation.z = std::copysign(std::sqrt(2. - 2. * a) * 0.5, b);
    result.push_back(ros_transform);
  }
  return result;
}

}  // namespace combine_grids
