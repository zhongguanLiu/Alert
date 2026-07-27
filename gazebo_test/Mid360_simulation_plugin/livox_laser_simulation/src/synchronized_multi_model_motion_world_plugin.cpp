#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <boost/bind/bind.hpp>
#include <gazebo/common/Events.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <ros/advertise_service_options.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <std_srvs/Trigger.h>
#include <XmlRpcValue.h>

namespace gazebo
{
namespace
{
constexpr int kFirstControlledId = 1;
constexpr int kLastControlledId = 10;
constexpr int kFirstFixedId = 11;
constexpr int kLastFixedId = 25;
constexpr int kFixedWalkingId = 6;
constexpr double kMillimetersToMeters = 0.001;
constexpr double kVectorTolerance = 1.0e-9;

bool XmlNumber(const XmlRpc::XmlRpcValue &_value, double &_result)
{
  if (_value.getType() == XmlRpc::XmlRpcValue::TypeInt)
  {
    _result = static_cast<int>(_value);
    return true;
  }
  if (_value.getType() == XmlRpc::XmlRpcValue::TypeDouble)
  {
    _result = static_cast<double>(_value);
    return true;
  }
  return false;
}

bool XmlInteger(const XmlRpc::XmlRpcValue &_value, int &_result)
{
  double numeric = 0.0;
  if (!XmlNumber(_value, numeric) || !std::isfinite(numeric) ||
      std::fabs(numeric - std::round(numeric)) > 1.0e-9)
  {
    return false;
  }
  _result = static_cast<int>(std::llround(numeric));
  return true;
}

bool XmlVector3(const XmlRpc::XmlRpcValue &_value,
                ignition::math::Vector3d &_result)
{
  if (_value.getType() != XmlRpc::XmlRpcValue::TypeArray || _value.size() != 3)
    return false;

  double values[3] = {0.0, 0.0, 0.0};
  for (int index = 0; index < 3; ++index)
  {
    if (!XmlNumber(_value[index], values[index]) ||
        !std::isfinite(values[index]))
    {
      return false;
    }
  }
  _result.Set(values[0], values[1], values[2]);
  return true;
}

std::string JsonEscape(const std::string &_value)
{
  std::ostringstream output;
  for (const char character : _value)
  {
    switch (character)
    {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default: output << character; break;
    }
  }
  return output.str();
}
}  // namespace

class SynchronizedMultiModelMotionWorldPlugin final : public WorldPlugin
{
public:
  SynchronizedMultiModelMotionWorldPlugin() = default;

  ~SynchronizedMultiModelMotionWorldPlugin() override
  {
    this->alive_.store(false);
    this->callbackQueue_.disable();
    if (this->callbackThread_.joinable())
      this->callbackThread_.join();

    std::lock_guard<std::mutex> lock(this->mutex_);
    this->updateConnection_.reset();
    this->rosNode_.reset();
  }

  void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf) override
  {
    this->world_ = std::move(_world);
    if (_sdf && _sdf->HasElement("ros_namespace"))
      this->rosNamespace_ = _sdf->Get<std::string>("ros_namespace");

    std::string discoveryError;
    this->environmentValid_ = this->DiscoverEnvironment(discoveryError);
    if (this->environmentValid_)
    {
      this->ApplyPhysicsOwnership();
      gzmsg << "Synchronized motion plugin owns object IDs 1-10; "
            << "object IDs 11-25 are fixed.\n";
    }
    else
    {
      this->lastError_ = discoveryError;
      gzerr << "Synchronized motion plugin disabled: " << discoveryError << "\n";
    }

    this->updateConnection_ = event::Events::ConnectWorldUpdateBegin(
        std::bind(&SynchronizedMultiModelMotionWorldPlugin::OnUpdate, this,
                  std::placeholders::_1));

    if (!ros::isInitialized())
    {
      this->lastError_ =
          "ROS is not initialized; launch Gazebo through gazebo_ros before control";
      gzerr << this->lastError_ << "\n";
      return;
    }

    this->rosNode_ = std::make_unique<ros::NodeHandle>(this->rosNamespace_);
    this->statusPublisher_ =
        this->rosNode_->advertise<std_msgs::String>("status", 1, true);

    this->reloadService_ = this->AdvertiseService(
        "reload", &SynchronizedMultiModelMotionWorldPlugin::Reload);
    this->startService_ = this->AdvertiseService(
        "start", &SynchronizedMultiModelMotionWorldPlugin::Start);
    this->stopService_ = this->AdvertiseService(
        "stop", &SynchronizedMultiModelMotionWorldPlugin::Stop);
    this->statusService_ = this->AdvertiseService(
        "get_status", &SynchronizedMultiModelMotionWorldPlugin::GetStatus);

    this->alive_.store(true);
    this->callbackThread_ = std::thread([this]()
    {
      while (this->alive_.load() && ros::ok())
        this->callbackQueue_.callAvailable(ros::WallDuration(0.05));
    });

    std::lock_guard<std::mutex> lock(this->mutex_);
    this->PublishStatus();
  }

private:
  enum class State
  {
    Unconfigured,
    Armed,
    Waiting,
    Running,
    Holding,
    Completed,
    Stopped,
    Error
  };

  struct ModelHandle
  {
    physics::ModelPtr model;
    physics::LinkPtr driveLink;
  };

  struct MotionCommand
  {
    int id = 0;
    std::string modelName;
    std::string frame;
    ignition::math::Vector3d linear;
    ignition::math::Vector3d angular;
  };

  using ServiceCallback = bool (SynchronizedMultiModelMotionWorldPlugin::*)(
      std_srvs::Trigger::Request &, std_srvs::Trigger::Response &);

  ros::ServiceServer AdvertiseService(const std::string &_name,
                                      ServiceCallback _callback)
  {
    ros::AdvertiseServiceOptions options =
        ros::AdvertiseServiceOptions::create<std_srvs::Trigger>(
            _name,
            boost::bind(_callback, this, boost::placeholders::_1,
                        boost::placeholders::_2),
            ros::VoidPtr(), &this->callbackQueue_);
    return this->rosNode_->advertiseService(options);
  }

  bool DiscoverEnvironment(std::string &_error)
  {
    std::map<int, physics::ModelPtr> modelsById;
    for (const physics::ModelPtr &model : this->world_->Models())
    {
      std::set<int> modelIds;
      for (const physics::LinkPtr &link : model->GetLinks())
      {
        for (const physics::CollisionPtr &collision : link->GetCollisions())
        {
          const double retro = collision->GetLaserRetro();
          if (!std::isfinite(retro) || retro <= 0.0)
            continue;
          const int id = static_cast<int>(std::llround(retro));
          if (id < 1 || id > 254 || std::fabs(retro - id) > 1.0e-6)
          {
            _error = model->GetName() + " has a non-integer laser_retro";
            return false;
          }
          modelIds.insert(id);
        }
      }

      if (modelIds.empty())
        continue;
      if (modelIds.size() != 1)
      {
        _error = model->GetName() + " has multiple laser_retro IDs";
        return false;
      }

      const int id = *modelIds.begin();
      if (modelsById.count(id) != 0)
      {
        _error = "laser_retro ID " + std::to_string(id) +
                 " is shared by multiple models";
        return false;
      }
      modelsById[id] = model;
    }

    for (int id = kFirstControlledId; id <= kLastFixedId; ++id)
    {
      if (modelsById.count(id) == 0)
      {
        _error = "missing environment object ID " + std::to_string(id);
        return false;
      }
    }

    for (int id = kFirstControlledId; id <= kLastControlledId; ++id)
    {
      physics::ModelPtr model = modelsById.at(id);
      std::vector<physics::LinkPtr> collisionLinks;
      for (const physics::LinkPtr &link : model->GetLinks())
      {
        if (!link->GetCollisions().empty())
          collisionLinks.push_back(link);
      }
      if (collisionLinks.size() != 1)
      {
        _error = model->GetName() + " (ID " + std::to_string(id) +
                 ") must have exactly one collision-bearing drive link";
        return false;
      }
      this->controlledModels_[id] = {model, collisionLinks.front()};
    }

    for (int id = kFirstFixedId; id <= kLastFixedId; ++id)
      this->fixedModels_[id] = modelsById.at(id);
    return true;
  }

  void ApplyPhysicsOwnership()
  {
    const ignition::math::Vector3d zero = ignition::math::Vector3d::Zero;
    for (auto &entry : this->controlledModels_)
    {
      const ModelHandle &handle = entry.second;
      handle.model->SetStatic(false);
      handle.model->SetGravityMode(false);
      for (const physics::LinkPtr &link : handle.model->GetLinks())
      {
        link->SetGravityMode(false);
        link->SetForce(zero);
        link->SetTorque(zero);
      }
      handle.driveLink->SetKinematic(true);
      handle.driveLink->SetEnabled(true);
      handle.driveLink->SetLinearVel(zero);
      handle.driveLink->SetAngularVel(zero);
    }

    for (auto &entry : this->fixedModels_)
    {
      const physics::ModelPtr &model = entry.second;
      model->SetLinearVel(zero);
      model->SetAngularVel(zero);
      model->SetGravityMode(false);
      for (const physics::LinkPtr &link : model->GetLinks())
      {
        link->SetForce(zero);
        link->SetTorque(zero);
      }
      model->SetStatic(true);
    }
  }

  bool ReadConfiguration(std::map<int, MotionCommand> &_commands,
                         double &_startDelay, double &_duration,
                         double &_endHold, double &_linearSpeedMmS,
                         double &_angularSpeedRadS,
                         double &_personWalkingSwitchTime,
                         ignition::math::Vector3d &_personWalkingSecondLinear,
                         std::string &_error) const
  {
    if (!this->rosNode_)
    {
      _error = "ROS node is unavailable";
      return false;
    }

    if (!this->rosNode_->getParam("start_delay", _startDelay) ||
        !std::isfinite(_startDelay) || _startDelay < 0.0)
    {
      _error = "start_delay must be a finite non-negative number";
      return false;
    }
    if (!this->rosNode_->getParam("duration", _duration) ||
        !std::isfinite(_duration))
    {
      _error = "duration must be finite; values <= 0 mean manual stop";
      return false;
    }
    if (!this->rosNode_->getParam("end_hold", _endHold) ||
        !std::isfinite(_endHold) || _endHold < 0.0)
    {
      _error = "end_hold must be a finite non-negative number";
      return false;
    }
    if (!this->rosNode_->getParam("linear_speed_mm_s", _linearSpeedMmS) ||
        !std::isfinite(_linearSpeedMmS) || _linearSpeedMmS < 0.0)
    {
      _error = "linear_speed_mm_s must be a finite non-negative number";
      return false;
    }
    if (!this->rosNode_->getParam("angular_speed_rad_s", _angularSpeedRadS) ||
        !std::isfinite(_angularSpeedRadS) || _angularSpeedRadS < 0.0)
    {
      _error = "angular_speed_rad_s must be a finite non-negative number";
      return false;
    }
    if (!this->rosNode_->getParam("person_walking_switch_time",
                                  _personWalkingSwitchTime) ||
        !std::isfinite(_personWalkingSwitchTime) ||
        _personWalkingSwitchTime < 0.0)
    {
      _error =
          "person_walking_switch_time must be a finite non-negative number";
      return false;
    }

    XmlRpc::XmlRpcValue secondWalkingLinearValue;
    if (!this->rosNode_->getParam("person_walking_second_linear_mps",
                                  secondWalkingLinearValue) ||
        !XmlVector3(secondWalkingLinearValue, _personWalkingSecondLinear) ||
        !std::isfinite(_personWalkingSecondLinear.Length()))
    {
      _error =
          "person_walking_second_linear_mps must contain 3 finite numbers";
      return false;
    }

    XmlRpc::XmlRpcValue models;
    if (!this->rosNode_->getParam("models", models) ||
        models.getType() != XmlRpc::XmlRpcValue::TypeArray)
    {
      _error = "models must be a YAML list";
      return false;
    }

    std::map<int, MotionCommand> parsed;
    for (int index = 0; index < models.size(); ++index)
    {
      XmlRpc::XmlRpcValue &item = models[index];
      if (item.getType() != XmlRpc::XmlRpcValue::TypeStruct ||
          !item.hasMember("id") || !item.hasMember("model_name") ||
          !item.hasMember("command_frame") ||
          !item.hasMember("linear_direction") ||
          !item.hasMember("angular_axis") ||
          !item.hasMember("fixed_linear_mps"))
      {
        _error = "every models entry requires id, model_name, command_frame, "
                 "linear_direction, angular_axis and fixed_linear_mps";
        return false;
      }

      MotionCommand command;
      if (!XmlInteger(item["id"], command.id) ||
          command.id < kFirstControlledId || command.id > kLastControlledId)
      {
        _error = "model id must be an integer from 1 through 10";
        return false;
      }
      if (parsed.count(command.id) != 0)
      {
        _error = "duplicate model id " + std::to_string(command.id);
        return false;
      }
      if (item["model_name"].getType() != XmlRpc::XmlRpcValue::TypeString ||
          item["command_frame"].getType() != XmlRpc::XmlRpcValue::TypeString)
      {
        _error = "model_name and command_frame must be strings";
        return false;
      }
      command.modelName = static_cast<std::string>(item["model_name"]);
      command.frame = static_cast<std::string>(item["command_frame"]);
      std::transform(command.frame.begin(), command.frame.end(),
                     command.frame.begin(), [](unsigned char character)
                     {
                       return static_cast<char>(std::tolower(character));
                     });
      if (command.frame != "world" && command.frame != "body")
      {
        _error = "command_frame must be world or body for ID " +
                 std::to_string(command.id);
        return false;
      }
      ignition::math::Vector3d linearDirection;
      ignition::math::Vector3d angularAxis;
      ignition::math::Vector3d fixedLinear;
      if (!XmlVector3(item["linear_direction"], linearDirection) ||
          !XmlVector3(item["angular_axis"], angularAxis) ||
          !XmlVector3(item["fixed_linear_mps"], fixedLinear))
      {
        _error = "linear_direction, angular_axis and fixed_linear_mps must "
                 "each contain 3 numbers";
        return false;
      }
      const double linearDirectionLength = linearDirection.Length();
      const double angularAxisLength = angularAxis.Length();
      if ((linearDirectionLength > kVectorTolerance &&
           std::fabs(linearDirectionLength - 1.0) > kVectorTolerance) ||
          (angularAxisLength > kVectorTolerance &&
           std::fabs(angularAxisLength - 1.0) > kVectorTolerance))
      {
        _error = "linear_direction and angular_axis must be zero or unit "
                 "vectors for ID " + std::to_string(command.id);
        return false;
      }
      if (linearDirectionLength > kVectorTolerance &&
          fixedLinear.Length() > kVectorTolerance)
      {
        _error = "ID " + std::to_string(command.id) +
                 " cannot combine experiment and fixed linear speeds";
        return false;
      }
      if (command.id == kFixedWalkingId)
      {
        const ignition::math::Vector3d expectedWalkingVelocity(0.0, -0.1, 0.0);
        const ignition::math::Vector3d expectedSecondWalkingVelocity(
            0.0, 0.1, 0.0);
        if (command.frame != "world" ||
            linearDirectionLength > kVectorTolerance ||
            angularAxisLength > kVectorTolerance ||
            !fixedLinear.Equal(expectedWalkingVelocity, kVectorTolerance) ||
            !_personWalkingSecondLinear.Equal(expectedSecondWalkingVelocity,
                                               kVectorTolerance))
        {
          _error =
              "ID 6 person_walking must use world -Y 0.1 m/s then +Y 0.1 m/s";
          return false;
        }
      }
      command.linear =
          linearDirection * (_linearSpeedMmS * kMillimetersToMeters) +
          fixedLinear;
      command.angular = angularAxis * _angularSpeedRadS;

      const ModelHandle &handle = this->controlledModels_.at(command.id);
      if (command.modelName != handle.model->GetName())
      {
        _error = "ID " + std::to_string(command.id) + " resolves to " +
                 handle.model->GetName() + ", not " + command.modelName;
        return false;
      }
      parsed[command.id] = command;
    }

    if (parsed.size() != static_cast<std::size_t>(kLastControlledId))
    {
      _error = "models must contain every controlled ID from 1 through 10";
      return false;
    }
    _commands = std::move(parsed);
    return true;
  }

  bool Reload(std_srvs::Trigger::Request &,
              std_srvs::Trigger::Response &_response)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    if (!this->environmentValid_)
    {
      _response.success = false;
      _response.message = this->lastError_;
      return true;
    }
    if (this->state_ == State::Waiting || this->state_ == State::Running ||
        this->state_ == State::Holding)
    {
      _response.success = false;
      _response.message = "stop the active plan before reloading";
      return true;
    }

    std::map<int, MotionCommand> parsed;
    double startDelay = 0.0;
    double duration = 0.0;
    double endHold = 0.0;
    double linearSpeedMmS = 1.0;
    double angularSpeedRadS = 0.0015;
    double personWalkingSwitchTime = 20.0;
    ignition::math::Vector3d personWalkingSecondLinear =
        ignition::math::Vector3d(0.0, 0.1, 0.0);
    std::string error;
    if (!this->ReadConfiguration(parsed, startDelay, duration, endHold,
                                 linearSpeedMmS, angularSpeedRadS,
                                 personWalkingSwitchTime,
                                 personWalkingSecondLinear, error))
    {
      this->lastError_ = error;
      this->state_ = State::Error;
      _response.success = false;
      _response.message = error;
      this->PublishStatus();
      return true;
    }

    this->commands_ = std::move(parsed);
    this->startDelay_ = startDelay;
    this->duration_ = duration;
    this->endHold_ = endHold;
    this->linearSpeedMmS_ = linearSpeedMmS;
    this->angularSpeedRadS_ = angularSpeedRadS;
    this->personWalkingSwitchTime_ = personWalkingSwitchTime;
    this->personWalkingSecondLinear_ = personWalkingSecondLinear;
    this->startRequested_ = false;
    this->stopRequested_ = false;
    this->scheduledStartTime_ = std::numeric_limits<double>::quiet_NaN();
    this->actualStartTime_ = std::numeric_limits<double>::quiet_NaN();
    this->holdStartTime_ = std::numeric_limits<double>::quiet_NaN();
    this->state_ = State::Armed;
    this->lastError_.clear();
    _response.success = true;
    _response.message = this->BuildStatusJson();
    this->PublishStatus();
    return true;
  }

  bool Start(std_srvs::Trigger::Request &,
             std_srvs::Trigger::Response &_response)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    if (this->state_ != State::Armed && this->state_ != State::Completed &&
        this->state_ != State::Stopped)
    {
      _response.success = false;
      _response.message = "reload a valid complete plan before start";
      return true;
    }
    this->startRequested_ = true;
    this->stopRequested_ = false;
    _response.success = true;
    _response.message = "start queued for one atomic Gazebo update";
    return true;
  }

  bool Stop(std_srvs::Trigger::Request &,
            std_srvs::Trigger::Response &_response)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->stopRequested_ = true;
    this->startRequested_ = false;
    _response.success = true;
    _response.message = "stop queued for one atomic Gazebo update";
    return true;
  }

  bool GetStatus(std_srvs::Trigger::Request &,
                 std_srvs::Trigger::Response &_response)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    _response.success = this->environmentValid_ && this->state_ != State::Error;
    _response.message = this->BuildStatusJson();
    return true;
  }

  void OnUpdate(const common::UpdateInfo &_info)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    const double now = _info.simTime.Double();
    if (!std::isfinite(now) || !this->environmentValid_)
      return;

    if (std::isfinite(this->lastSimTime_) && now < this->lastSimTime_)
    {
      this->ZeroControlledModels();
      this->startRequested_ = false;
      this->stopRequested_ = false;
      if (!this->commands_.empty())
        this->state_ = State::Armed;
      this->scheduledStartTime_ = std::numeric_limits<double>::quiet_NaN();
      this->actualStartTime_ = std::numeric_limits<double>::quiet_NaN();
      this->holdStartTime_ = std::numeric_limits<double>::quiet_NaN();
      this->lastError_ = "simulation time moved backwards; explicit restart required";
      this->PublishStatus();
    }
    this->lastSimTime_ = now;

    if (this->stopRequested_)
    {
      this->stopRequested_ = false;
      this->startRequested_ = false;
      this->ZeroControlledModels();
      this->state_ = State::Stopped;
      this->PublishStatus();
      return;
    }

    if (this->startRequested_)
    {
      this->startRequested_ = false;
      this->scheduledStartTime_ = now + this->startDelay_;
      this->actualStartTime_ = std::numeric_limits<double>::quiet_NaN();
      this->holdStartTime_ = std::numeric_limits<double>::quiet_NaN();
      this->state_ = State::Waiting;
      this->PublishStatus();
    }

    if (this->state_ == State::Waiting)
    {
      this->ZeroControlledModels();
      if (now + 1.0e-12 < this->scheduledStartTime_)
        return;
      this->actualStartTime_ = now;
      this->state_ = State::Running;
      this->PublishStatus();
    }

    if (this->state_ == State::Running)
    {
      if (this->duration_ > 0.0 &&
          now - this->actualStartTime_ >= this->duration_ - 1.0e-12)
      {
        this->ZeroControlledModels();
        if (this->endHold_ > 0.0)
        {
          this->holdStartTime_ = now;
          this->state_ = State::Holding;
        }
        else
        {
          this->state_ = State::Completed;
        }
        this->PublishStatus();
        return;
      }
      this->ApplyCommands(now - this->actualStartTime_);
      return;
    }

    if (this->state_ == State::Holding)
    {
      this->ZeroControlledModels();
      if (now - this->holdStartTime_ >= this->endHold_ - 1.0e-12)
      {
        this->state_ = State::Completed;
        this->PublishStatus();
      }
      return;
    }

    this->ZeroControlledModels();
  }

  void ApplyCommands(const double _runningTime)
  {
    const ignition::math::Vector3d zero = ignition::math::Vector3d::Zero;
    for (const auto &entry : this->commands_)
    {
      const MotionCommand &command = entry.second;
      const ModelHandle &handle = this->controlledModels_.at(command.id);
      ignition::math::Vector3d linear = command.linear;
      ignition::math::Vector3d angular = command.angular;
      if (command.id == kFixedWalkingId &&
          _runningTime >= this->personWalkingSwitchTime_)
      {
        linear = this->personWalkingSecondLinear_;
      }
      if (command.frame == "body")
      {
        const ignition::math::Quaterniond rotation =
            handle.model->WorldPose().Rot();
        linear = rotation.RotateVector(linear);
        angular = rotation.RotateVector(angular);
      }
      handle.driveLink->SetForce(zero);
      handle.driveLink->SetTorque(zero);
      handle.driveLink->SetLinearVel(linear);
      handle.driveLink->SetAngularVel(angular);
    }
  }

  void ZeroControlledModels()
  {
    const ignition::math::Vector3d zero = ignition::math::Vector3d::Zero;
    for (auto &entry : this->controlledModels_)
    {
      const ModelHandle &handle = entry.second;
      handle.driveLink->SetForce(zero);
      handle.driveLink->SetTorque(zero);
      handle.driveLink->SetLinearVel(zero);
      handle.driveLink->SetAngularVel(zero);
    }
  }

  std::string StateName() const
  {
    switch (this->state_)
    {
      case State::Unconfigured: return "UNCONFIGURED";
      case State::Armed: return "ARMED";
      case State::Waiting: return "WAITING";
      case State::Running: return "RUNNING";
      case State::Holding: return "HOLDING";
      case State::Completed: return "COMPLETED";
      case State::Stopped: return "STOPPED";
      case State::Error: return "ERROR";
    }
    return "ERROR";
  }

  std::string BuildStatusJson() const
  {
    std::ostringstream output;
    output.precision(12);
    output << "{\"state\":\"" << this->StateName() << "\""
           << ",\"environment_valid\":"
           << (this->environmentValid_ ? "true" : "false")
           << ",\"sim_time\":";
    if (std::isfinite(this->lastSimTime_))
      output << this->lastSimTime_;
    else
      output << "null";
    output << ",\"scheduled_start_time\":";
    if (std::isfinite(this->scheduledStartTime_))
      output << this->scheduledStartTime_;
    else
      output << "null";
    output << ",\"actual_start_time\":";
    if (std::isfinite(this->actualStartTime_))
      output << this->actualStartTime_;
    else
      output << "null";
    output << ",\"hold_start_time\":";
    if (std::isfinite(this->holdStartTime_))
      output << this->holdStartTime_;
    else
      output << "null";
    output << ",\"start_delay\":" << this->startDelay_
           << ",\"duration\":" << this->duration_
           << ",\"end_hold\":" << this->endHold_
           << ",\"linear_speed_mm_s\":" << this->linearSpeedMmS_
           << ",\"angular_speed_rad_s\":" << this->angularSpeedRadS_
           << ",\"person_walking_switch_time\":"
           << this->personWalkingSwitchTime_
           << ",\"controlled_ids\":[1,2,3,4,5,6,7,8,9,10]"
           << ",\"last_error\":\"" << JsonEscape(this->lastError_) << "\"}";
    return output.str();
  }

  void PublishStatus()
  {
    if (!this->statusPublisher_)
      return;
    std_msgs::String message;
    message.data = this->BuildStatusJson();
    this->statusPublisher_.publish(message);
  }

  physics::WorldPtr world_;
  event::ConnectionPtr updateConnection_;
  std::map<int, ModelHandle> controlledModels_;
  std::map<int, physics::ModelPtr> fixedModels_;
  std::map<int, MotionCommand> commands_;

  std::string rosNamespace_ = "/synchronized_model_motion";
  std::unique_ptr<ros::NodeHandle> rosNode_;
  ros::CallbackQueue callbackQueue_;
  ros::ServiceServer reloadService_;
  ros::ServiceServer startService_;
  ros::ServiceServer stopService_;
  ros::ServiceServer statusService_;
  ros::Publisher statusPublisher_;
  std::thread callbackThread_;
  std::atomic<bool> alive_{false};

  mutable std::mutex mutex_;
  State state_ = State::Unconfigured;
  bool environmentValid_ = false;
  bool startRequested_ = false;
  bool stopRequested_ = false;
  double startDelay_ = 0.0;
  double duration_ = 0.0;
  double endHold_ = 0.0;
  double linearSpeedMmS_ = 0.0;
  double angularSpeedRadS_ = 0.0;
  double personWalkingSwitchTime_ = 20.0;
  ignition::math::Vector3d personWalkingSecondLinear_{0.0, 0.1, 0.0};
  double lastSimTime_ = std::numeric_limits<double>::quiet_NaN();
  double scheduledStartTime_ = std::numeric_limits<double>::quiet_NaN();
  double actualStartTime_ = std::numeric_limits<double>::quiet_NaN();
  double holdStartTime_ = std::numeric_limits<double>::quiet_NaN();
  std::string lastError_;
};

GZ_REGISTER_WORLD_PLUGIN(SynchronizedMultiModelMotionWorldPlugin)
}  // namespace gazebo
