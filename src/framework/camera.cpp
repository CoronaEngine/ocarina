//
// Created by Zero on 06/06/2022.
//

#include "camera.h"

namespace ocarina {

math3d::Matrix4 Camera::get_view_matrix() {
    update_matrices();
    return view_matrix_;
}

math3d::Matrix4 Camera::get_projection_matrix() {
    update_matrices();
    return projection_matrix_;
}

void Camera::calculate_view_matrix() {
    view_matrix_ = math3d::Matrix4::look_at(position_, target_, up_);
}

void Camera::calculate_projection_matrix() {
    projection_matrix_ = math3d::Matrix4::perspective(math3d::deg_to_rad(fov_), aspect_ratio_, znear_, zfar_);//transform::perspective<H>(fov_, znear_, zfar_);
}

}// namespace ocarina