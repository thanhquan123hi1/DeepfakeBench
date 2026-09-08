import os
import sys
import logging

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import DETECTOR

logger = logging.getLogger(__name__)

def _safe_import(import_fn):
    try:
        import_fn()
    except Exception as e:
        logger.debug(f"Optional detector not loaded: {e}")

_safe_import(lambda: __import__('training.detectors.utils.slowfast', fromlist=['slowfast']))
_safe_import(lambda: __import__('training.detectors.facexray_detector', fromlist=['FaceXrayDetector']))
_safe_import(lambda: __import__('training.detectors.xception_detector', fromlist=['XceptionDetector']))
_safe_import(lambda: __import__('training.detectors.efficientnetb4_detector', fromlist=['EfficientDetector']))
_safe_import(lambda: __import__('training.detectors.resnet34_detector', fromlist=['ResnetDetector']))
_safe_import(lambda: __import__('training.detectors.f3net_detector', fromlist=['F3netDetector']))
_safe_import(lambda: __import__('training.detectors.meso4_detector', fromlist=['Meso4Detector']))
_safe_import(lambda: __import__('training.detectors.meso4Inception_detector', fromlist=['Meso4InceptionDetector']))
_safe_import(lambda: __import__('training.detectors.spsl_detector', fromlist=['SpslDetector']))
_safe_import(lambda: __import__('training.detectors.core_detector', fromlist=['CoreDetector']))
_safe_import(lambda: __import__('training.detectors.capsule_net_detector', fromlist=['CapsuleNetDetector']))
_safe_import(lambda: __import__('training.detectors.srm_detector', fromlist=['SRMDetector']))
_safe_import(lambda: __import__('training.detectors.ucf_detector', fromlist=['UCFDetector']))
_safe_import(lambda: __import__('training.detectors.recce_detector', fromlist=['RecceDetector']))
_safe_import(lambda: __import__('training.detectors.fwa_detector', fromlist=['FWADetector']))
_safe_import(lambda: __import__('training.detectors.ffd_detector', fromlist=['FFDDetector']))
_safe_import(lambda: __import__('training.detectors.videomae_detector', fromlist=['VideoMAEDetector']))
_safe_import(lambda: __import__('training.detectors.clip_detector', fromlist=['CLIPDetector']))
_safe_import(lambda: __import__('training.detectors.timesformer_detector', fromlist=['TimeSformerDetector']))
_safe_import(lambda: __import__('training.detectors.xclip_detector', fromlist=['XCLIPDetector']))
_safe_import(lambda: __import__('training.detectors.sbi_detector', fromlist=['SBIDetector']))
_safe_import(lambda: __import__('training.detectors.ftcn_detector', fromlist=['FTCNDetector']))
_safe_import(lambda: __import__('training.detectors.i3d_detector', fromlist=['I3DDetector']))
_safe_import(lambda: __import__('training.detectors.altfreezing_detector', fromlist=['AltFreezingDetector']))
_safe_import(lambda: __import__('training.detectors.stil_detector', fromlist=['STILDetector']))
_safe_import(lambda: __import__('training.detectors.lsda_detector', fromlist=['LSDADetector']))
_safe_import(lambda: __import__('training.detectors.sladd_detector', fromlist=['SLADDXceptionDetector']))
_safe_import(lambda: __import__('training.detectors.pcl_xception_detector', fromlist=['PCLXceptionDetector']))
_safe_import(lambda: __import__('training.detectors.iid_detector', fromlist=['IIDDetector']))
_safe_import(lambda: __import__('training.detectors.lrl_detector', fromlist=['LRLDetector']))
_safe_import(lambda: __import__('training.detectors.rfm_detector', fromlist=['RFMDetector']))
_safe_import(lambda: __import__('training.detectors.uia_vit_detector', fromlist=['UIAViTDetector']))
_safe_import(lambda: __import__('training.detectors.multi_attention_detector', fromlist=['MultiAttentionDetector']))
_safe_import(lambda: __import__('training.detectors.sia_detector', fromlist=['SIADetector']))
_safe_import(lambda: __import__('training.detectors.tall_detector', fromlist=['TALLDetector']))
_safe_import(lambda: __import__('training.detectors.effort_detector', fromlist=['EffortDetector']))
_safe_import(lambda: __import__('training.detectors.gend_detector', fromlist=['GenDDetector']))
_safe_import(lambda: __import__('training.detectors.gend_effort_detector', fromlist=['GenDEffortDetector']))
_safe_import(lambda: __import__('training.detectors.effort_asy', fromlist=['EffortAsyDetector']))
_safe_import(lambda: __import__('training.detectors.BiasLoraAsy', fromlist=['BiasLoraAsyDetector']))
_safe_import(lambda: __import__('training.detectors.clip_bias_detector', fromlist=['CLIPBiasDetector']))
