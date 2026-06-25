from .custom_architectures import (
    ObsCaudalLocalActJointIndependentActionHead,
    ObsCaudalLocalVelActJointIndependentActionHead,
    ObsDriveFbActDriveFbSplitPartitionedActionHead,
    ObsDriveFbCurriculumActDriveFbSplitPartitionedActionHead,
    ObsDriveFbHistoryActDriveFbSplitPartitionedActionHead,
    ObsGlobalActJointIndependentActionHead,
    ObsGlobalActSplitIndependentActionHead,
    ObsGlobalActThreeRegionIndependentActionHead,
    ObsGlobalExtendedActThreeRegionIndependentActionHead,
    ObsLocalActJointIndependentActionHead,
    ObsLocalActJointSharedActionHead,
    ObsRegionActFrontBackIndependentActionHead,
    ObsRegionActFrontBackSharedActionHead,
    ObsRegionActThreeRegionIndependentActionHead,
    ObsRegionActThreeRegionPartialSharedActionHead,
    ObsWindow3ActJointIndependentActionHead,
    ObsWindow3ActJointPartialSharedActionHead,
    ObsWindow3VelActJointPartialSharedActionHead,
    ObsWindow5ActJointIndependentActionHead,
    ObsWindow5ActJointPartialSharedActionHead,
    ObsWindow7ActJointIndependentActionHead,
    ObsWindow7ActThreeRegionIndependentActionHead,
)
from .extractors import (
    GroupedStateHistoryExtractor,
    PerFeatureStateHistoryExtractor,
    StandardConfigurableExtractor,
)


NETWORK_REGISTRY = {
    # Config key -> descriptive class. Comments keep the previous class names.
    "shared": ObsLocalActJointSharedActionHead,  # localFeedbackShared
    "non-shared": ObsLocalActJointIndependentActionHead,  # localFeedbackNonShared
    "nn3": ObsGlobalActSplitIndependentActionHead,
    "nn4": ObsRegionActFrontBackIndependentActionHead,
    "nn5": ObsRegionActFrontBackSharedActionHead,
    "nn6": ObsWindow3ActJointPartialSharedActionHead,
    "nn7": ObsWindow3ActJointIndependentActionHead,
    "caudl": ObsCaudalLocalVelActJointIndependentActionHead,
    "caudl2": ObsCaudalLocalActJointIndependentActionHead,
    "nn8": ObsGlobalActThreeRegionIndependentActionHead,
    "nn9": ObsRegionActThreeRegionIndependentActionHead,
    "enn1": ObsGlobalActJointIndependentActionHead,
    "enn2": ObsGlobalExtendedActThreeRegionIndependentActionHead,
    "enn3": ObsWindow7ActJointIndependentActionHead,
    "enn4": ObsWindow5ActJointIndependentActionHead,
    "enn5": ObsWindow7ActThreeRegionIndependentActionHead,
    "enn6": ObsWindow5ActJointPartialSharedActionHead,
    "enn7": ObsWindow3VelActJointPartialSharedActionHead,
    "enn8": ObsRegionActThreeRegionPartialSharedActionHead,
    "dnn1": ObsDriveFbActDriveFbSplitPartitionedActionHead,
    "dnn2": ObsDriveFbCurriculumActDriveFbSplitPartitionedActionHead,
    "dnn3": ObsDriveFbHistoryActDriveFbSplitPartitionedActionHead,
}

NETWORK_ALIASES = {
    # Very local position/phase controllers.
    "obs_local__act_joint__shared": "shared",
    "obs_local__act_joint__independent": "non-shared",
    # Caudal-local controllers are per-joint in the current implementation.
    "obs_caudal_local__act_joint__independent": "caudl2",
    "obs_caudal_local_vel__act_joint__independent": "caudl",
    # Sliding local windows. The window number is the number of neighboring
    # body positions represented per position/phase/velocity group.
    "obs_window3__act_joint__partial_shared": "nn6",
    "obs_window3__act_joint__independent": "nn7",
    "obs_window3_vel__act_joint__partial_shared": "enn7",
    "obs_window5__act_joint__independent": "enn4",
    "obs_window5__act_joint__partial_shared": "enn6",
    "obs_window7__act_joint__independent": "enn3",
    # Regional action heads.
    "obs_region__act_frontback__independent": "nn4",
    "obs_region__act_frontback__shared": "nn5",
    "obs_region__act_three_region__independent": "nn9",
    "obs_region__act_three_region__partial_shared": "enn8",
    "obs_window7__act_three_region__independent": "enn5",
    # Global observation heads.
    "obs_global__act_split__independent": "nn3",
    "obs_global__act_three_region__independent": "nn8",
    "obs_global_extended__act_three_region__independent": "enn2",
    "obs_global__act_joint__independent": "enn1",
    # Drive/feedback split heads.
    "obs_drive_fb__act_drive_fb_split__partitioned": "dnn1",
    "obs_drive_fb_curriculum__act_drive_fb_split__partitioned": "dnn2",
    "obs_drive_fb_history__act_drive_fb_split__partitioned": "dnn3",
}

STATE_HISTORY_REGISTRY = {
    "sh1": PerFeatureStateHistoryExtractor,
    "sh2": GroupedStateHistoryExtractor,
}


def _valid_local_feedback_message() -> str:
    old_names = ", ".join(sorted(NETWORK_REGISTRY))
    readable_aliases = ", ".join(sorted(NETWORK_ALIASES))
    return f"valid old names: {old_names}; valid readable aliases: {readable_aliases}"

# Legacy aliases for reading old experiment configs, checkpoints, and notes.
CustomNetwork = StandardConfigurableExtractor
sh1 = PerFeatureStateHistoryExtractor
sh2 = GroupedStateHistoryExtractor
localFeedbackShared = ObsLocalActJointSharedActionHead
localFeedbackNonShared = ObsLocalActJointIndependentActionHead
nn3 = ObsGlobalActSplitIndependentActionHead
nn4 = ObsRegionActFrontBackIndependentActionHead
nn5 = ObsRegionActFrontBackSharedActionHead
nn6 = ObsWindow3ActJointPartialSharedActionHead
nn7 = ObsWindow3ActJointIndependentActionHead
caudl = ObsCaudalLocalVelActJointIndependentActionHead
caudl2 = ObsCaudalLocalActJointIndependentActionHead
enn1 = ObsGlobalActJointIndependentActionHead
enn2 = ObsGlobalExtendedActThreeRegionIndependentActionHead
enn3 = ObsWindow7ActJointIndependentActionHead
enn4 = ObsWindow5ActJointIndependentActionHead
enn5 = ObsWindow7ActThreeRegionIndependentActionHead
enn6 = ObsWindow5ActJointPartialSharedActionHead
enn7 = ObsWindow3VelActJointPartialSharedActionHead
enn8 = ObsRegionActThreeRegionPartialSharedActionHead
nn8 = ObsGlobalActThreeRegionIndependentActionHead
nn9 = ObsRegionActThreeRegionIndependentActionHead
dnn1 = ObsDriveFbActDriveFbSplitPartitionedActionHead
dnn2 = ObsDriveFbCurriculumActDriveFbSplitPartitionedActionHead
dnn3 = ObsDriveFbHistoryActDriveFbSplitPartitionedActionHead
