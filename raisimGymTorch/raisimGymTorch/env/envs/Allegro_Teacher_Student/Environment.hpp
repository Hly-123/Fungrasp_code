

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include "raisim/World.hpp"
#include <vector>
#include "raisim/math.hpp"
#include <math.h>


#include<iostream>
#include <fstream>
#include <string>
using namespace std;

# define CONTACT_RATIO


namespace raisim {
    class ENVIRONMENT : public RaisimGymEnv {

    public:

        explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {


            world_ = std::make_unique<raisim::World>();
            world_->addGround();
            world_->setERP(0.0);

            std::string hand_model =  cfg["hand_model"].As<std::string>();
            thresd_cos = cfg["thresd_cos"].As<double>();
            closer = cfg["closer"].As<double>();
            opener = cfg["opener"].As<double>();
            tighter = cfg["tighter"].As<double>();
            exp_log = cfg["exp_log"].As<bool>();
            obj_weight_scale = cfg["obj_weight_scale"].As<double>();
            if (exp_log)
            {
                RSWARN("!!!!!!!!!!!!!!!!!!!!!!!!!!!!START to RECORD!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                store_file_name =  cfg["store_file_name"].As<std::string>();
                ofstream outFile(store_file_name, ios::out);
                if (outFile.is_open())
                {
                    outFile << cfg["obj"].As<std::string>() 
                    << ',' << "rewards_.sum()"
                    << ',' << "fingertip_closer_reward * 1.0"
                    << ',' << "root_pose_reward_ * 0.1"
                    << ',' << "grasp_reward * 1.0"
                    << ',' << "pos_reward * 2.0"
                    << ',' << "pose_reward * 0.1"
                    << ',' << "contact_reward * 1.0"
                    << ',' << "impulse_reward * 2.0"
                    << ',' << "rel_obj_reward_ * -1.0"
                    << ',' << "body_vel_reward_ * -0.5"
                    << ',' << "body_qvel_reward_ * -0.5"
                    << ',' << "Metric1-Geodesic-obj_reward_"
                    << ',' << "Metric2-MPE-obj_pose_reward_"
                    << ',' << "Metric3-Simulated_Dist"
                    << endl;
                }
                else{cout << "file is failed to open" << endl;}
                outFile.close();
            }
            

            resourceDir_ = resourceDir;
            mano_ = world_->addArticulatedSystem(resourceDir+"/AllegroHand/"+hand_model,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63));
            mano_->setName("mano");


            box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
            box->setPosition(1.25, 0, 0.25);
            box->setAppearance("0.0 0.0 0.0 0.0");


            mano_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);


            gcDim_ = mano_->getGeneralizedCoordinateDim();
            gvDim_ = mano_->getDOF();
            nJoints_ = gcDim_-3;


            gc_.setZero(gcDim_);
            gv_.setZero(gvDim_);
            gc_set_.setZero(gcDim_); gv_set_.setZero(gvDim_);

            
            reward_obj_pose.setZero(3);

            pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget6_.setZero(6);
            final_pose_.setZero(nJoints_), final_obj_pos_.setZero(7), final_ee_pos_.setZero(num_bodyparts*3), final_contact_pos_.setZero(num_contacts*3), final_vertex_normals_.setZero(num_contacts*3), contact_body_idx_.setZero(num_contacts), final_contact_array_.setZero(num_contacts);
            rel_pose_.setZero(nJoints_), rel_obj_pos_.setZero(3), rel_objpalm_pos_.setZero(3), rel_body_pos_.setZero(num_bodyparts*3), rel_contact_pos_.setZero(num_contacts*3),  rel_obj_pose_.setZero(3), contacts_.setZero(num_contacts), rel_contacts_.setZero(num_contacts), impulses_.setZero(num_contacts);
            actionDim_ = gcDim_; actionMean_.setZero(actionDim_);  actionStd_.setOnes(actionDim_);
            joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);
            Position.setZero(); Obj_Position.setZero(); Rel_fpos.setZero(); Obj_orientation.setZero();
            obj_quat.setZero();obj_quat[0] = 1.0;
            Obj_linvel.setZero(); Obj_qvel.setZero();
            rel_obj_vel.setZero(); rel_obj_qvel.setZero();
            rel_body_table_pos_.setZero();
            bodyLinearVel_.setZero(); bodyAngularVel_.setZero();
            init_root_.setZero(); init_or_.setZero(); init_rot_.setZero();
            obj_pose_.setZero(); obj_pos_init_.setZero(7);
            Fpos_world.setZero();
            final_obj_pos_[3] = 1.0;


            final_ee_pos_origin.setZero(num_bodyparts*3);
            current_ee_pos.setZero(num_bodyparts*3);
            contact_current_ee_pos.setZero(num_bodyparts*3);
            contact_final_ee_pos.setZero(num_bodyparts*3);

            last_action.setZero(gcDim_);


            finger_weights_.setOnes(num_bodyparts*3);
            finger_weights_.segment(4*3,3) *= 4;
            finger_weights_.segment(8*3,3) *= 4;
            finger_weights_.segment(12*3,3) *= 4;
            finger_weights_.segment(16*3,3) *= 6;
            finger_weights_ /= finger_weights_.sum();
            finger_weights_ *= num_bodyparts*3;


            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.head(3).setConstant(50);
            jointDgain.head(3).setConstant(0.1);
            jointPgain.tail(nJoints_).setConstant(50.0);
            jointDgain.tail(nJoints_).setConstant(0.2);
            

            mano_->setPdGains(jointPgain, jointDgain);
            mano_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
            mano_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));


            obDim_double = gcDim_ + gvDim_ + final_ee_pos_.size() + final_pose_.head(3).size() + num_contacts*3 + rel_objpalm_pos_.size() + rel_obj_pose_.size() + rel_obj_pos_.size() + rel_obj_vel.size() + rel_obj_qvel.size() + rel_body_table_pos_.size() + 4;
            hand_dim = gcDim_ + 3*2 + rel_body_pos_.size() + rel_pose_.head(3).size() + num_contacts;
            obj_dim = rel_objpalm_pos_.size() + rel_obj_pose_.size() + rel_obj_pos_.size() + rel_obj_vel.size() + rel_obj_qvel.size() + rel_body_table_pos_.size();


            tobeEncode_dim = last_action.size() + gcDim_ + rel_pose_.head(3).size() + num_contacts;

            obDim_double += tobeEncode_dim;
            obDouble_.setZero(obDim_double);

            ob_delay.setZero(obDim_double);

            history_len = 10;
            concat_dim = tobeEncode_dim*history_len + obDim_double;
            ob_concat.setZero(concat_dim);
            obDim_ = concat_dim;  

            root_guided =  cfg["root_guided"].As<bool>();
            float finger_action_std = cfg["finger_action_std"].As<float>();
            float rot_action_std = cfg["rot_action_std"].As<float>();


            joint_limits_ = mano_->getJointLimits();
            

            for(int i=0;i < int(gcDim_); i++){
                actionMean_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                joint_limit_low[i] = joint_limits_[i][0];
                joint_limit_high[i] = joint_limits_[i][1];
            }


            if (root_guided){
                actionStd_.setConstant(finger_action_std);
                actionStd_.head(3).setConstant(0.001);
                actionStd_.segment(3,3).setConstant(rot_action_std);
            }
            else{
                actionStd_.setConstant(finger_action_std);
                actionStd_.head(3).setConstant(0.01);
                actionStd_.segment(3,3).setConstant(0.01);
            }


            world_->setMaterialPairProp("object", "object", 0.9, 0.0, 0.0);
            world_->setMaterialPairProp("object", "finger", 0.9, 0.0, 0.0);
            world_->setMaterialPairProp("finger", "finger", 0.8, 0.0, 0.0);


            rewards_.initializeFromConfigurationFile (cfg["reward"]);


            if (visualizable_) {
                if(server_) server_->lockVisualizationServerMutex();
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer(8080);


                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                table_top->setPosition(1.25, 0, 0.475);
    
                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                leg1->setPosition(0.2625,0.4675,0.2375);
                leg2->setPosition(2.2275,0.4875,0.2375);
                leg3->setPosition(0.2625,-0.4675,0.2375);
                leg4->setPosition(2.2275,-0.4875,0.2375);


                for(int i=0; i<num_bodyparts ; i++){
                    spheres[i] = server_->addVisualSphere(body_parts_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                }
                if(server_) server_->unlockVisualizationServerMutex();
            }

        }

        void init() final { }


        void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {


            raisim::Mat<3, 3> inertia;
            inertia.setIdentity();
            const raisim::Vec<3> com = {0, 0, 0};
            
            obj_weight_ = obj_weight[0];

            box_obj_mesh = false;
            cylinder_mesh = false;

            if (world_->getObject("cylinder") != NULL)
                world_->removeObject(cylinder);
            if (world_->getObject("box_obj") != NULL)
                world_->removeObject(box_obj);
            if (world_->getObject("mesh_obj") != NULL)
                world_->removeObject(obj_mesh_1);
            if (visualizable_ && world_->getObject("mesh_obj_2") != NULL)
                world_->removeObject(obj_mesh_2);

            if (obj_type[0] == 0)
            {
                cylinder = static_cast<raisim::Cylinder*>(world_->addCylinder(obj_dim[0],obj_dim[1],obj_weight_,"object", raisim::COLLISION(2)));
                cylinder->setCom(com);
                cylinder->setInertia(inertia);
                cylinder->setName("cylinder");
                cylinder->setPosition(obj_pos_init_[0],obj_pos_init_[1],obj_pos_init_[2]);
                obj_idx_ = cylinder->getIndexInWorld();
                cylinder->setOrientation(1,0,0,0);
                cylinder->setVelocity(0,0,0,0,0,0);
                cylinder_mesh = true;

            }

            else if (obj_type[0] == 1)
            {
                box_obj = static_cast<raisim::Box*>(world_->addBox(obj_dim[0],obj_dim[1],obj_dim[2], obj_weight_,"object", raisim::COLLISION(2)));
                box_obj->setCom(com);
                box_obj->setInertia(inertia);
                box_obj->setName("box_obj");
                box_obj->setPosition(obj_pos_init_[0],obj_pos_init_[1],obj_pos_init_[2]);
                obj_idx_ = box_obj->getIndexInWorld();
                box_obj->setOrientation(1,0,0,0);
                box_obj->setVelocity(0,0,0,0,0,0);
                box_obj_mesh = true;
            }


            else
            {
                std::string obj_name;

                if (obj_type[0] == 3)
                    obj_name = resourceDir_ + "/meshes_simplified/" + ycb_objects_[obj_idx[0]] + "/mesh_aligned.obj";

                else
                    obj_name = resourceDir_ + "/meshes_simplified/" + ycb_objects_[obj_idx[0]] + "/textured_meshlab_quart.obj";
                obj_mesh_1 =  static_cast<raisim::Mesh*>(world_->addMesh(obj_name, obj_weight_, inertia, com, 1.0,"object",raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(63)));
                obj_idx_ = obj_mesh_1->getIndexInWorld();
                obj_mesh_1->setPosition(obj_pos_init_[0],obj_pos_init_[1],obj_pos_init_[2]);
                obj_mesh_1->setOrientation(1,0,0,0);
                obj_mesh_1->setVelocity(0,0,0,0,0,0);
                obj_mesh_1->setName("mesh_obj");
            }


            if (visualizable_)
            {
                std::string obj_name_target = resourceDir_ + "/meshes_simplified/" + ycb_objects_[obj_idx[0]] + "/textured_meshlab.obj";
                obj_mesh_2 =  static_cast<raisim::Mesh*>(world_->addMesh(obj_name_target, obj_weight_, inertia, com, 1.0,"object",raisim::COLLISION(10),raisim::COLLISION(10)));


                obj_mesh_2->setAppearance("0 1 0 0.5");
                obj_mesh_2->setName("mesh_obj_2");
            }


        }


        void reset() final {

            if (first_reset_)
            {
                first_reset_=false;
            }
            else{

                actionMean_.setZero();
                mano_->setState(gc_set_, gv_set_);
                if (cylinder_mesh)
                {
                    cylinder->setPosition(obj_pos_init_[0],obj_pos_init_[1],obj_pos_init_[2]);
                    cylinder->setOrientation(obj_pos_init_[3],obj_pos_init_[4],obj_pos_init_[5],obj_pos_init_[6]);
                    cylinder->setVelocity(0,0,0,0,0,0);
                }
                else if (box_obj_mesh)
                {
                    box_obj->setPosition(obj_pos_init_[0],obj_pos_init_[1],obj_pos_init_[2]);
                    box_obj->setOrientation(obj_pos_init_[3],obj_pos_init_[4],obj_pos_init_[5],obj_pos_init_[6]);
                    box_obj->setVelocity(0,0,0,0,0,0);
                }
                else{
                    obj_mesh_1->setPosition(obj_pos_init_[0],obj_pos_init_[1],obj_pos_init_[2]);
                    obj_mesh_1->setOrientation(obj_pos_init_[3],obj_pos_init_[4],obj_pos_init_[5],obj_pos_init_[6]);
                    obj_mesh_1->setVelocity(0,0,0,0,0,0);
                }

                box->clearExternalForcesAndTorques();
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                updateObservation();
                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_->setGeneralizedForce(gen_force);
            }

        }


        void reset_state(const Eigen::Ref<EigenVec>& init_state, const Eigen::Ref<EigenVec>& init_vel, const Eigen::Ref<EigenVec>& obj_pose) final {


            obs_history.clear();

            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);


            jointPgain.head(3).setConstant(80);

            jointDgain.head(3).setConstant(0.1);
            jointPgain.tail(nJoints_).setConstant(60.0);
            jointDgain.tail(nJoints_).setConstant(0.2);


            mano_->setPdGains(jointPgain, jointDgain);


            Eigen::VectorXd gen_force;
            gen_force.setZero(gcDim_);
            mano_->setGeneralizedForce(gen_force);


            box->setPosition(1.25, 0, 0.25);
            box->setOrientation(1,0,0,0);
            box->setVelocity(0,0,0,0,0,0);

            gc_set_.head(6).setZero();
            gc_set_.tail(nJoints_-3) = init_state.tail(nJoints_-3).cast<double>();
            gv_set_ = init_vel.cast<double>();
            mano_->setState(gc_set_, gv_set_);


            init_root_ = init_state.head(3);
            mano_->setBasePos(init_state.head(3));


            raisim::Vec<4> quat;
            raisim::eulerToQuat(init_state.segment(3,3),quat);
            raisim::quatToRotMat(quat,init_rot_);
            raisim::transpose(init_rot_,init_or_);
            mano_->setBaseOrientation(init_rot_);


            obj_pos_init_  = obj_pose.cast<double>();

            if (cylinder_mesh)
            {
                cylinder->setPosition(obj_pos_init_[0],obj_pos_init_[1],obj_pos_init_[2]);
                cylinder->setOrientation(obj_pos_init_[3],obj_pos_init_[4],obj_pos_init_[5],obj_pos_init_[6]);
                cylinder->setVelocity(0,0,0,0,0,0);
            }
            else if (box_obj_mesh)
            {
                box_obj->setPosition(obj_pos_init_[0],obj_pos_init_[1],obj_pos_init_[2]);
                box_obj->setOrientation(obj_pos_init_[3],obj_pos_init_[4],obj_pos_init_[5],obj_pos_init_[6]);
                box_obj->setVelocity(0,0,0,0,0,0);

            }
            else{
                obj_mesh_1->setPosition(obj_pos_init_[0],obj_pos_init_[1],obj_pos_init_[2]);
                obj_mesh_1->setOrientation(obj_pos_init_[3],obj_pos_init_[4],obj_pos_init_[5],obj_pos_init_[6]);
                obj_mesh_1->setVelocity(0,0,0,0,0,0);
            }


            actionMean_.setZero();
            actionMean_.tail(nJoints_-3) = gc_set_.tail(nJoints_-3);

            motion_synthesis = false;
            root_guiding_counter_= 0;
            obj_table_contact_ = 0;


            gen_force.setZero(gcDim_);
            mano_->setGeneralizedForce(gen_force);
            updateObservation();
        }


        void set_goals(const Eigen::Ref<EigenVec>& obj_goal_pos, const Eigen::Ref<EigenVec>& ee_goal_pos, const Eigen::Ref<EigenVec>& goal_pose, const Eigen::Ref<EigenVec>& contact_pos, const Eigen::Ref<EigenVec>& goal_contacts) final {

            raisim::Vec<4> quat_goal_hand_w, quat_goal_hand_r, quat_obj_init;
            raisim::Vec<3> euler_goal_pose;
            raisim::Mat<3,3> rotm_goal_hand_r;


            final_ee_pos_origin = ee_goal_pos.cast<double>();

            final_obj_pos_ = obj_goal_pos.cast<double>();


            raisim::quatToRotMat(obj_goal_pos.tail(4), Obj_orientation_temp);
            quat_obj_init = obj_goal_pos.tail(4);
            raisim::transpose(Obj_orientation_temp,Obj_orientation);

            raisim::eulerToQuat(goal_pose.head(3),quat_goal_hand_w);
            raisim::quatToRotMat(quat_goal_hand_w,root_pose_world_);


            raisim::quatInvQuatMul(quat_obj_init,quat_goal_hand_w,quat_goal_hand_r);
            raisim::quatToRotMat(quat_goal_hand_r,rotm_goal_hand_r);
            raisim::RotmatToEuler(rotm_goal_hand_r,euler_goal_pose);


            final_pose_ = goal_pose.cast<double>();
            final_pose_.head(3) = euler_goal_pose.e();


            for(int i=0; i<num_bodyparts; i++){


                if (i==0)
                {
                    Position[0] = ee_goal_pos[i*3]-final_obj_pos_[0];
                    Position[1] = ee_goal_pos[i*3+1]-final_obj_pos_[1];
                    Position[2] = ee_goal_pos[i*3+2]-final_obj_pos_[2];
                }

                else
                {
                    Position[0] = ee_goal_pos[i*3]-final_obj_pos_[0];
                    Position[1] = ee_goal_pos[i*3+1]-final_obj_pos_[1];
                    Position[2] = ee_goal_pos[i*3+2]-final_obj_pos_[2];
                }


                raisim::matvecmul(Obj_orientation,Position,Rel_fpos);


                final_ee_pos_[i*3] = Rel_fpos[0];
                final_ee_pos_[i*3+1] = Rel_fpos[1];
                final_ee_pos_[i*3+2] =  Rel_fpos[2];

            }


            num_active_contacts_ = float(goal_contacts.sum());
            final_contact_array_ = goal_contacts.cast<double>();

            for(int i=0; i<num_contacts ; i++){
                contact_body_idx_[i] =  mano_->getBodyIdx(contact_bodies_[i]);
                contactMapping_.insert(std::pair<int,int>(int(mano_->getBodyIdx(contact_bodies_[i])),i));
            }


            k_contact = 1.0/num_active_contacts_;

        }


        float step(const Eigen::Ref<EigenVec>& action) final {

            raisim::Vec<4> obj_orientation_quat, quat_final_pose, quat_world;
            raisim::Mat<3, 3> rot, rot_trans, rot_world, rot_goal, rotmat_final_obj_pos, rotmat_final_obj_pos_trans;
            raisim::Vec<3> obj_pos_raisim, euler_goal_world, final_obj_pose_mat, hand_pos_world, hand_pose, act_pos, act_or_pose;
            raisim::transpose(Obj_orientation_temp,Obj_orientation);
            obj_pos_raisim[0] = final_obj_pos_[0]-Obj_Position[0]; obj_pos_raisim[1] = final_obj_pos_[1]-Obj_Position[1]; obj_pos_raisim[2] = final_obj_pos_[2]-Obj_Position[2];


            if (motion_synthesis)
            {
                raisim::quatToRotMat(final_obj_pos_.tail(4),rotmat_final_obj_pos);
                raisim::transpose(rotmat_final_obj_pos, rotmat_final_obj_pos_trans);

                raisim::matvecmul(rotmat_final_obj_pos, final_ee_pos_.head(3), Fpos_world);

                raisim::vecadd(obj_pos_raisim, Fpos_world);


                if (visualizable_)
                {
                    spheres[0]->setPosition(pos_goal.e());
                }


                pos_goal = action.head(3);

                raisim::vecsub(pos_goal, init_root_, act_pos);

                raisim::matvecmul(init_or_,obj_pos_raisim,act_or_pose);


                actionMean_.head(3) = (act_or_pose.e())*std::min(1.0,(0.008*root_guiding_counter_));
                actionMean_.head(3)  += gc_.head(3);

                raisim::Mat<3, 3> rotmat_gc, rotmat_gc_trans, rotmat_obj_pose, posegoal_rotmat;
                raisim::Vec<4> quat_gc, quat_pg, quat_diff;
                raisim::Vec<3> euler_obj_pose_goal, euler_obj_pose_curr, diff_obj_pose, rot_goal_euler;
                pose_goal = action.segment(3,3);
                raisim::eulerToQuat(pose_goal,quat_pg);

                raisim::quatToRotMat(final_obj_pos_.tail(4),rotmat_gc);
                raisim::matmul(init_or_, rotmat_gc, rotmat_gc_trans);
                raisim::RotmatToEuler(rotmat_gc_trans, euler_obj_pose_goal);

                raisim::matmul(init_or_, Obj_orientation_temp, rotmat_obj_pose);
                raisim::RotmatToEuler(rotmat_obj_pose, euler_obj_pose_curr);

                raisim::vecsub(euler_obj_pose_goal, euler_obj_pose_curr, diff_obj_pose);


                for (int i = 0; i < 3; i++) {
                    if (diff_obj_pose[i] > pi_)
                        diff_obj_pose[i] -= 2*pi_;
                    else if (diff_obj_pose[i] < -pi_)
                        diff_obj_pose[i] += 2*pi_;
                }


                actionMean_.segment(3,3) = rel_obj_pose_ * std::min(1.0,(0.0008*root_guiding_counter_));
                actionMean_.segment(3,3) += gc_.segment(3,3);
                root_guiding_counter_ += 1;

            }


            else if (root_guided){

                if (cylinder_mesh)
                {
                    Obj_Position = cylinder->getPosition();
                    Obj_orientation_temp = cylinder->getRotationMatrix();
                    cylinder->getQuaternion(obj_quat);
                }
                else if (box_obj_mesh)
                {
                    Obj_Position = box_obj->getPosition();
                    Obj_orientation_temp = box_obj->getRotationMatrix();
                    box_obj->getQuaternion(obj_quat);
                }
                else
                {
                    Obj_Position = obj_mesh_1->getPosition();
                    Obj_orientation_temp = obj_mesh_1->getRotationMatrix();
                    obj_mesh_1->getQuaternion(obj_quat);
                }


                raisim::matvecmul(Obj_orientation_temp, final_ee_pos_.head(3), Fpos_world);
                raisim::vecadd(Obj_Position, Fpos_world);

                raisim::vecsub(Fpos_world, init_root_, act_pos);
                raisim::matvecmul(init_or_,act_pos,act_or_pose);

                actionMean_.head(3) = act_or_pose.e();


            }


            pTarget_ = action.cast<double>();


            pTarget_ = pTarget_.cwiseProduct(actionStd_);
            pTarget_ += actionMean_;


            Eigen::VectorXd pTarget_clipped;
            pTarget_clipped.setZero(gcDim_);
            pTarget_clipped = pTarget_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

            last_action = pTarget_clipped.cast<double>();


            mano_->setPdTarget(pTarget_clipped, vTarget_);


            for(int i=0; i<  int(control_dt_ / simulation_dt_ + 1e-10); i++){
                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();
                if(server_) server_->unlockVisualizationServerMutex();
            }


            updateObservation();
            actionMean_ = gc_;

            pose_reward_ = -(rel_pose_).norm();
            root_pos_reward_ = -rel_body_pos_.head(3).norm();   
            root_pose_reward_ = -rel_pose_.head(3).squaredNorm();


            pos_reward_ = -rel_body_pos_.cwiseProduct(finger_weights_).squaredNorm();
            obj_reward_ = -rel_obj_pos_.norm();
            contact_pos_reward_ =  rel_contact_pos_.squaredNorm();
            obj_pose_reward_ = -reward_obj_pose.squaredNorm();


            grasp_reward = (cos_simularity).sum() * 0.125;

            if(grasp_reward >= (thresd_cos-1))
            {
                grasp_reward *= 0.1;
            }
            else{grasp_reward *= 0.2;}


            contact_current_ee_pos_squareNorm = contact_current_ee_pos.cwiseProduct(finger_weights_).squaredNorm();
            contact_final_ee_pos_squareNorm = contact_final_ee_pos.cwiseProduct(finger_weights_).squaredNorm();
            if (contact_current_ee_pos_squareNorm >= contact_final_ee_pos_squareNorm && contact_current_ee_pos_squareNorm != 0 && (!isnan(contact_current_ee_pos_squareNorm)))
            {
                contact_weight = contact_final_ee_pos_squareNorm / contact_current_ee_pos_squareNorm;
            }
            else{contact_weight = 1.0;}
            # ifdef CONTACT_RATIO
            contact_weight = 1.0;
            # endif
            contact_reward_ = contact_weight * k_contact*(rel_contacts_.sum());


            rel_obj_reward_ = rel_obj_vel.squaredNorm();
            body_vel_reward_ = bodyLinearVel_.squaredNorm();
            body_qvel_reward_ = bodyAngularVel_.squaredNorm();


            impulse_reward_ = ((final_contact_array_.cwiseProduct(impulses_)).sum());


            if (current_gripper_distance > target_gripper_distance && contact_reward_ <= 0)
            {
                fingertip_closer_reward = closer * (target_gripper_distance - current_gripper_distance);
            }

            else if (current_gripper_distance <= target_gripper_distance && contact_reward_ <= 0)
            {
                fingertip_closer_reward = opener * (current_gripper_distance - target_gripper_distance);
            }

            else if (current_gripper_distance <= target_gripper_distance && contact_reward_ > 0)
            {
                fingertip_closer_reward = contact_reward_ * (1+(target_gripper_distance - current_gripper_distance));
            }

            else if (current_gripper_distance > target_gripper_distance && contact_reward_ > 0)
            {
                fingertip_closer_reward = tighter*(contact_reward_ * (1+(current_gripper_distance - target_gripper_distance)));
            }


            if(isnan(impulse_reward_))
                impulse_reward_ = 0.0;
        

            rewards_.record("fingertip_closer_reward", std::max(-10.0, fingertip_closer_reward));
            rewards_.record("grasp_reward", std::max(-10.0, grasp_reward));

            rewards_.record("pos_reward", std::max(-10.0, pos_reward_));
            rewards_.record("root_pos_reward_", std::max(-10.0, root_pos_reward_));
            rewards_.record("root_pose_reward_", std::max(-10.0, root_pose_reward_));

            rewards_.record("pose_reward", std::max(-10.0, pose_reward_));
            rewards_.record("contact_pos_reward", std::max(-10.0, contact_pos_reward_));

            rewards_.record("contact_reward", std::max(-10.0, contact_reward_));
            rewards_.record("obj_reward", std::max(-10.0, obj_reward_));
            rewards_.record("obj_pose_reward_", std::max(-10.0,obj_pose_reward_));
            rewards_.record("impulse_reward", std::min(impulse_reward_, obj_weight_*5));

            rewards_.record("rel_obj_reward_", std::max(0.0, rel_obj_reward_));

            rewards_.record("body_vel_reward_", std::max(0.0,body_vel_reward_));

            rewards_.record("body_qvel_reward_", std::max(0.0,body_qvel_reward_));
            rewards_.record("torque", std::max(0.0, mano_->getGeneralizedForce().squaredNorm()));
            

            num_step += 1;

            
            if (num_step == 60)
            {
                if (exp_log)
                {
                    simulated_dist = max(-10.0,obj_pose_reward_);
                }
            }
            
            if (num_step == 245)
            {
                num_step = 0;
                num_exp += 1;
                if (exp_log)
                {
                    RSWARN("!!!!!!!!!!!!!!!!!!!!!!!!!!!!START to RECORD!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                    ofstream addFile(store_file_name, ios::app);
                    if (addFile.is_open())
                    {

                        addFile << to_string(num_exp)
                                << ',' << to_string(rewards_.sum())
                                << ',' << to_string(fingertip_closer_reward*1.0)
                                << ',' << to_string(max(-10.0, root_pose_reward_)*0.1)
                                << ',' << to_string(max(-10.0, grasp_reward)*1.0)
                                << ',' << to_string(max(-10.0, pos_reward_)*2.0) 
                                << ',' << to_string(max(-10.0, pose_reward_)*0.1)
                                << ',' << to_string(max(-10.0, contact_reward_)*1.0)
                                << ',' << to_string(min(impulse_reward_, obj_weight_*5)*2.0)
                                << ',' << to_string(max(0.0, rel_obj_reward_)*(-1.0))
                                << ',' << to_string(max(0.0,body_vel_reward_)*(-0.5))
                                << ',' << to_string(max(0.0,body_qvel_reward_)*(-0.5))
                                << ',' << to_string(max(-10.0, angle_metric))
                                << ',' << to_string(max(-10.0,obj_pose_reward_))
                                << ',' << to_string(simulated_dist)
                                << endl;
                    }
                    addFile.close();
                }
                
            }


            # ifdef CONTACT_RATIO
            return contact_reward_;
            # else  
            return rewards_.sum();
            # endif
        }


        void updateObservation() {
            raisim::Vec<4> quat, quat_hand, quat_obj_init;
            raisim::Vec<3> body_vel, obj_frame_diff, obj_frame_diff_w, obj_frame_diff_h, euler_hand, sphere_pos, norm_pos, rel_wbody_root, euler_obj, rel_rbody_root, rel_body_table, rel_obj_init, rel_objpalm, rel_obj_pose_r3;
            raisim::Mat<3,3> rot, rot_mult, body_orientation_transpose, palm_world_pose_mat, palm_world_pose_mat_trans, obj_pose_wrist_mat, rel_pose_mat, final_obj_rotmat_temp, diff_obj_pose_mat, final_obj_wrist, obj_wrist, obj_wrist_trans, final_obj_pose_mat;


            raisim::Vec<3> sphere_pos_origin;

            contacts_.setZero();
            rel_contacts_.setZero();
            impulses_.setZero();


            mano_->getState(gc_, gv_);


            if (cylinder_mesh)
            {
                Obj_Position = cylinder->getPosition();
                Obj_orientation_temp = cylinder->getRotationMatrix();
                cylinder->getQuaternion(obj_quat);
                Obj_qvel = cylinder->getAngularVelocity();
                Obj_linvel = cylinder->getLinearVelocity();
            }
            else if (box_obj_mesh)
            {
                Obj_Position = box_obj->getPosition();
                Obj_orientation_temp = box_obj->getRotationMatrix();
                box_obj->getQuaternion(obj_quat);
                Obj_qvel = box_obj->getAngularVelocity();
                Obj_linvel = box_obj->getLinearVelocity();
            }
            else
            {
                Obj_Position = obj_mesh_1->getPosition();
                Obj_orientation_temp = obj_mesh_1->getRotationMatrix();
                obj_mesh_1->getQuaternion(obj_quat);
                Obj_qvel = obj_mesh_1->getAngularVelocity();
                Obj_linvel = obj_mesh_1->getLinearVelocity();
            }
            raisim::transpose(Obj_orientation_temp, Obj_orientation);


            rel_pose_ = final_pose_-gc_.tail(gcDim_-3);


            mano_->getFrameOrientation(body_parts_[0], palm_world_pose_mat);
            raisim::transpose(palm_world_pose_mat,palm_world_pose_mat_trans);

            raisim::matmul(palm_world_pose_mat_trans, Obj_orientation_temp, obj_pose_wrist_mat);
            raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_);


            for(int i=0; i< num_bodyparts ; i++){
                mano_->getFramePosition(body_parts_[i], Position);
                mano_->getFrameOrientation(body_parts_[i], Body_orientation);

                if (i==0)
                {
                    raisim::transpose(Body_orientation, body_orientation_transpose);
                    rel_objpalm[0] = Position[0]-Obj_Position[0];
                    rel_objpalm[1] = Position[1]-Obj_Position[1];
                    rel_objpalm[2] = Position[2]-Obj_Position[2];

                    rel_objpalm_pos_ = Body_orientation.e().transpose()*rel_objpalm.e();

                    rel_body_table[0] = 0.0;
                    rel_body_table[1] = 0.0;
                    rel_body_table[2] = Position[2]-0.5;
                    rel_body_table_pos_ = Body_orientation.e().transpose()*rel_body_table.e();


                    rel_obj_init[0] = obj_pos_init_[0] - Obj_Position[0];
                    rel_obj_init[1] = obj_pos_init_[1] - Obj_Position[1];
                    rel_obj_init[2] = obj_pos_init_[2] - Obj_Position[2];


                    rel_obj_pos_ = Body_orientation.e().transpose()*rel_obj_init.e();

                    raisim::matmul(Obj_orientation, Body_orientation, rot_mult);
                    raisim::RotmatToEuler(rot_mult,euler_hand);

                    rel_pose_.head(3) = final_pose_.head(3) - euler_hand.e();

                    bodyLinearVel_ =  gv_.segment(0, 3);
                    bodyAngularVel_ = gv_.segment(3, 3);

                    rel_obj_vel = Body_orientation.e().transpose() * Obj_linvel.e();
                    rel_obj_qvel = Body_orientation.e().transpose() * Obj_qvel.e();

                    raisim::quatToRotMat(final_obj_pos_.segment(3,4),final_obj_pose_mat);

                    raisim::matmul(init_or_, final_obj_pose_mat, final_obj_wrist);
                    raisim::matmul(init_or_, Obj_orientation_temp, obj_wrist);
                    raisim::transpose(obj_wrist, obj_wrist_trans);

                    raisim::matmul(final_obj_wrist, obj_wrist_trans, diff_obj_pose_mat);
                    raisim::RotmatToEuler(diff_obj_pose_mat,rel_obj_pose_r3);
                    rel_obj_pose_ = rel_obj_pose_r3.e();
                    

                }


                Position[0] = Position[0]-Obj_Position[0];
                Position[1] = Position[1]-Obj_Position[1];
                Position[2] = Position[2]-Obj_Position[2];
                raisim::matvecmul(Obj_orientation,Position,Rel_fpos);


                current_ee_pos[i * 3] = Rel_fpos[0];
                current_ee_pos[i * 3 + 1] = Rel_fpos[1];
                current_ee_pos[i * 3 + 2] = Rel_fpos[2];


                obj_frame_diff[0] = final_ee_pos_[i * 3]- Rel_fpos[0];
                obj_frame_diff[1] = final_ee_pos_[i * 3 + 1] - Rel_fpos[1];
                obj_frame_diff[2] = final_ee_pos_[i * 3 + 2] - Rel_fpos[2];
                raisim::matvecmul(Obj_orientation_temp,obj_frame_diff,obj_frame_diff_w);
                raisim::matvecmul(body_orientation_transpose,obj_frame_diff_w,obj_frame_diff_h);
                rel_body_pos_[i * 3] = obj_frame_diff_h[0];
                rel_body_pos_[i * 3 + 1] = obj_frame_diff_h[1];
                rel_body_pos_[i * 3 + 2] = obj_frame_diff_h[2];


                if (visualizable_)
                {
                    raisim::matvecmul(Obj_orientation_temp,{final_ee_pos_[i * 3], final_ee_pos_[i * 3 + 1], final_ee_pos_[i * 3 + 2]},sphere_pos);
                    vecadd(Obj_Position, sphere_pos);


                    spheres[i]->setPosition(sphere_pos.e());


                }

            }


            Eigen::Vector3d target_pose_point, current_pose_point, target_wrist_point, current_wrist_point;
            Eigen::Vector3d target_joint_direction, current_joint_direction;
            double cos_dot;
            Eigen::VectorXd match_joint_idex(8);
            Eigen::VectorXd contact_joint_idex(4);
            cos_simularity.setZero(8);

            target_wrist_point = final_ee_pos_.head(3);
            current_wrist_point = current_ee_pos.head(3);


            match_joint_idex << 9, 21, 33, 45, 12, 24, 36, 48;

            for(int i=0; i < match_joint_idex.size(); i++)
            {
                target_pose_point = final_ee_pos_.segment(match_joint_idex(i),3);
                current_pose_point = current_ee_pos.segment(match_joint_idex(i),3);

                if(i >= 4)
                {
                    target_wrist_point = final_ee_pos_.segment(match_joint_idex(i-4),3);
                    current_wrist_point = current_ee_pos.segment(match_joint_idex(i-4),3);
                }

                target_joint_direction = target_pose_point - target_wrist_point;
                current_joint_direction = current_pose_point - current_wrist_point;


                cos_dot = target_joint_direction.dot(current_joint_direction);
                cos_simularity(i) = (cos_dot / (target_joint_direction.norm()*current_joint_direction.norm()))-1;
            }
            

            Eigen::Vector3d target_gripper_thumb, target_gripper_finger, current_gripper_thumb, current_gripper_finger;


            target_gripper_thumb = final_ee_pos_.segment(48,3);
            target_gripper_finger = (final_ee_pos_.segment(12,3) + final_ee_pos_.segment(24,3) + final_ee_pos_.segment(36,3)) / 3;
            current_gripper_thumb = current_ee_pos.segment(48,3);
            current_gripper_finger = (current_ee_pos.segment(12,3) + current_ee_pos.segment(24,3) + current_ee_pos.segment(36,3)) / 3;

            target_gripper_distance = (target_gripper_thumb - target_gripper_finger).norm();
            current_gripper_distance = (current_gripper_thumb - current_gripper_finger).norm();


            contact_current_ee_pos.segment(12,3) = current_ee_pos.segment(12,3);
            contact_current_ee_pos.segment(24,3) = current_ee_pos.segment(24,3);
            contact_current_ee_pos.segment(36,3) = current_ee_pos.segment(36,3);
            contact_current_ee_pos.segment(48,3) = current_ee_pos.segment(48,3);
            contact_final_ee_pos.segment(12,3) = final_ee_pos_.segment(12,3);
            contact_final_ee_pos.segment(24,3) = final_ee_pos_.segment(24,3);
            contact_final_ee_pos.segment(36,3) = final_ee_pos_.segment(36,3);
            contact_final_ee_pos.segment(48,3) = final_ee_pos_.segment(48,3);


            reward_obj_pose[0] = final_obj_pos_[0] - Obj_Position[0];
            reward_obj_pose[1] = final_obj_pos_[1] - Obj_Position[1];
            reward_obj_pose[2] = final_obj_pos_[2] - Obj_Position[2];


            dot_product = (final_obj_pos_.segment(3,4)).dot(obj_quat.e());
            angle_metric = std::acos(2 * dot_product * dot_product - 1);


            for(auto& contact: mano_->getContacts()) {
                if (contact.skip() || contact.getPairObjectIndex() != obj_idx_) continue;
                contacts_[contactMapping_[contact.getlocalBodyIndex()]] = 1;
                impulses_[contactMapping_[contact.getlocalBodyIndex()]] = contact.getImpulse().norm();
            }


            rel_contacts_ = final_contact_array_.cwiseProduct(contacts_);


            int obj_table_contact = 0;


            int table_idx = box->getIndexInWorld();
            if (cylinder_mesh) {
                for (auto &contact: cylinder->getContacts()) {
                    if (contact.skip() || contact.getPairObjectIndex() != table_idx)
                        continue;
                    obj_table_contact = 1;
                } 
            } else if (box_obj_mesh) {
                for (auto &contact: box_obj->getContacts()) {
                    if (contact.skip() || contact.getPairObjectIndex() != table_idx)
                        continue;
                    obj_table_contact = 1;
                }
            }
            else
            {
                for (auto &contact: obj_mesh_1->getContacts()) {
                    if (contact.skip() || contact.getPairObjectIndex() != table_idx)
                        continue;
                    obj_table_contact = 1;
                }
            }

            if (Obj_Position[2] < 0.1)
                obj_table_contact = 1;


            obDouble_ <<
                    gc_,
                    rel_pose_.head(3),
                    final_contact_array_,
                    last_action,
                    gc_,
                    bodyLinearVel_,
                    bodyAngularVel_,
                    gv_.tail(gvDim_ - 6),
                    rel_body_pos_,
                    rel_pose_.head(3),
                    rel_objpalm_pos_,
                    rel_obj_vel,
                    rel_obj_qvel,
                    final_contact_array_,
                    impulses_,
                    rel_contacts_,
                    rel_obj_pos_,
                    rel_body_table_pos_,
                    obj_pose_.e(),
                    Obj_Position.e(),
                    obj_table_contact;   


            obs_history.push_back(obDouble_);
        }


        void observe(Eigen::Ref<EigenVec> ob) final {


            int lag = 1;
            int vec_size = obs_history.size();   
            ob_delay << obs_history[vec_size - lag];

            if (vec_size == 1)
            {
                for (int i = 1; i < history_len; i++)
                {
                    obs_history.push_back(obDouble_);
                }
            }
            vec_size = obs_history.size();
            for (int i = 0; i < history_len; i++)
            {
                ob_concat.segment(tobeEncode_dim * i, tobeEncode_dim) << obs_history[vec_size - history_len + i].head(tobeEncode_dim);
            }
            ob_concat.tail(obDim_double) << ob_delay;
            ob = ob_concat.cast<float>();
        }


        void set_root_control() final {

            motion_synthesis = true;

            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);

            jointPgain.head(3).setConstant(500);
            jointDgain.head(3).setConstant(0.1);
            jointPgain.tail(nJoints_).setConstant(50.0);
            jointDgain.tail(nJoints_).setConstant(0.2);


            mano_->setPdGains(jointPgain, jointDgain);

            raisim::Vec<3> frame_pos;
            raisim::Mat<3,3> root_orientation, root_orientation_trans;

            mano_->getFrameOrientation(body_parts_[0], root_orientation);
            mano_->getFramePosition(body_parts_[0], frame_pos);
            raisim::transpose(root_orientation, root_orientation_trans);


            raisim::matvecmul(init_or_,{0, 0, obj_weight_*10*(1.0/obj_weight_scale)},up_gen_vec);
            gen_force_.setZero(gcDim_);
            gen_force_.head(3) = up_gen_vec.e();
            mano_->setGeneralizedForce(gen_force_);

            up_vec[0] = gc_[0];
            up_vec[1] = gc_[1];
            up_vec[2] = gc_[2];
            up_pose = gc_.segment(3,3);
            gc_set_ = gc_;

        }


        bool isTerminalState(float& terminalReward) final {

            if(obDouble_.hasNaN())
            {return true;}

            return false;
        }

    private:
        int gcDim_, gvDim_, nJoints_;
        bool visualizable_ = false;
        raisim::ArticulatedSystem* mano_;
        Eigen::VectorXd gc_, gv_, pTarget_, pTarget6_, vTarget_, gc_set_, gv_set_,  obj_pos_init_;
        Eigen::VectorXd gen_force_, final_obj_pos_, final_pose_, final_ee_pos_, final_contact_pos_, final_contact_array_, contact_body_idx_, final_vertex_normals_;
        double terminalRewardCoeff_ = -10.;
        double pose_reward_= 0.0;
        double pos_reward_ = 0.0;
        double contact_reward_= 0.0;
        double obj_reward_ = 0.0;
        double root_reward_ = 0.0;
        double contact_pos_reward_ = 0.0;
        double root_pos_reward_ = 0.0;
        double root_pose_reward_ = 0.0;
        double rel_obj_reward_ = 0.0;
        double body_vel_reward_ = 0.0;
        double body_qvel_reward_ = 0.0;
        double obj_pose_reward_ = 0.0;
        double falling_reward = 0.0;
        double k_obj = 50;
        double k_pose = 0.5;
        double k_ee = 1.0;
        double k_contact = 1.0;
        double ray_length = 0.05;
        double num_active_contacts_;
        double impulse_reward_ = 0.0;
        double obj_weight_;


        Eigen::VectorXd joint_limit_high, joint_limit_low, actionMean_, actionStd_, obDouble_, rel_pose_, finger_weights_, rel_obj_pos_, rel_objpalm_pos_, rel_body_pos_, rel_contact_pos_, rel_contacts_, contacts_, impulses_, rel_obj_pose_;

        Eigen::VectorXd reward_obj_pose;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, rel_obj_qvel, rel_obj_vel, up_pose, rel_body_table_pos_;
        std::set<size_t> footIndices_;
        raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
        raisim::Cylinder *cylinder;
        raisim::Box *box_obj;
        raisim::Box *box;
        int rf_dim = 6;
        int num_obj = 4;
        int num_contacts = 13;
        int num_bodyparts = 17;
        int obj_table_contact_ = 0;
        int root_guiding_counter_ = 0;
        int obj_idx_;
        bool root_guided=false;
        bool cylinder_mesh=false;
        bool box_obj_mesh=false;
        bool first_reset_=true;
        bool no_pose_state = false;
        bool nohierarchy = false;
        bool contact_pruned = false;
        bool motion_synthesis = false;
        raisim::Vec<3> pose_goal, pos_goal, up_vec, up_gen_vec, obj_pose_, Position, Obj_Position, Rel_fpos, Obj_linvel, Obj_qvel, Fpos_world, init_root_;
        raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Body_orientation, init_or_, root_pose_world_, init_rot_;
        raisim::Vec<4> obj_quat;
        std::vector<int> contact_idxs_;


        int num_step = 0;
        std::string store_file_name;
        int num_exp = 0;
        bool exp_log = false;
        double simulated_dist = 0;
        Eigen::VectorXd final_ee_pos_origin, current_ee_pos;
        Eigen::VectorXd cos_simularity;
        double grasp_reward;
        double thresd_cos;
        double target_gripper_distance, current_gripper_distance;
        double fingertip_closer_reward;
        double closer, opener, tighter;
        double dot_product, angle_metric;
        raisim::Vec<3> Start_Position;


        double contact_weight = 1.0;
        Eigen::VectorXd contact_final_ee_pos, contact_current_ee_pos;
        double contact_final_ee_pos_squareNorm, contact_current_ee_pos_squareNorm;


        Eigen::VectorXd last_action;

        std::deque<Eigen::VectorXd> obs_history;
        Eigen::VectorXd ob_delay,ob_concat;
        int concat_dim, tobeEncode_dim;
        int history_len;
        int hand_dim,obj_dim, obDim_double;


        double obj_weight_scale;
        

        std::string ORIGIN_body_parts_[21] =  {"WRJ0rz",
                                        "FFJ2", "FFJ1", "FFJ0","FFTip",
                                        "MFJ2", "MFJ1", "MFJ0","MFTip",
                                        "RFJ2", "RFJ1", "RFJ0","RFTip",
                                        "LFJ2", "LFJ1", "LFJ0","LFTip",
                                        "THJ2", "THJ1", "THJ0","THTip"
        };


        std::string body_parts_[17] = {"z_rotation_joint",
                                        "joint_1.0", "joint_2.0", "joint_3.0", "joint_3.0_tip",
                                        "joint_5.0", "joint_6.0", "joint_7.0", "joint_7.0_tip",
                                        "joint_9.0", "joint_10.0", "joint_11.0", "joint_11.0_tip",
                                        "joint_13.0", "joint_14.0", "joint_15.0", "joint_15.0_tip"};


        std::string contact_bodies_[13] = {"base_link",
                                             "link_1.0", "link_2.0", "link_3.0",
                                             "link_5.0", "link_6.0", "link_7.0",
                                             "link_9.0", "link_10.0", "link_11.0",
                                             "link_13.0", "link_14.0", "link_15.0"};


        std::string ORIGIN_contact_bodies_[16] =  {"link_palm_rz",
                                            "link_ff_pm_z", "link_ff_md_z", "link_ff_dd_z",
                                            "link_mf_pm_z", "link_mf_md_z", "link_mf_dd_z",
                                            "link_rf_pm_z", "link_rf_md_z", "link_rf_dd_z",
                                            "link_lf_pm_z", "link_lf_md_z", "link_lf_dd_z",
                                            "link_th_pm_z", "link_th_md_z", "link_th_dd_z"
        };


        std::string ycb_objects_[21] = {"002_master_chef_can",
                                        "003_cracker_box",
                                        "004_sugar_box",
                                        "005_tomato_soup_can",
                                        "006_mustard_bottle",
                                        "007_tuna_fish_can",
                                        "008_pudding_box",
                                        "009_gelatin_box",
                                        "010_potted_meat_can",
                                        "011_banana",
                                        "019_pitcher_base",
                                        "021_bleach_cleanser",
                                        "024_bowl",
                                        "025_mug",
                                        "035_power_drill",
                                        "036_wood_block",
                                        "037_scissors",
                                        "040_large_marker",
                                        "051_large_clamp",
                                        "052_extra_large_clamp",
                                        "061_foam_brick"};
                                        
        raisim::PolyLine *lines[21];
        raisim::Visuals *spheres[17];
        raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
        std::map<int,int> contactMapping_;
        std::string resourceDir_;
        std::vector<raisim::Vec<2>> joint_limits_;
        raisim::PolyLine *line;
        const double pi_ = 3.14159265358979323846;
    };
}