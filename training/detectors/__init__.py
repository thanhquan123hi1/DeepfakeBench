"""
Detector Registry Module
Fixed to avoid circular imports and use correct utils path
"""

from metrics.registry import Registry

DETECTOR = Registry('detector')

# Lazy loading to avoid circular imports
def _lazy_load_detectors():
    """Load all detector modules after DETECTOR registry is initialized."""
    try:
        from .gend_effort_detector import GenDEffortDetector
    except ImportError as e:
        pass

# Load detectors after DETECTOR registry is created
_lazy_load_detectors()
