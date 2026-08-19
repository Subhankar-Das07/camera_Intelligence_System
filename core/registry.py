from typing import Dict, Type
from core.base_pipeline import BaseVideoPipeline
from pipelines.intrusion_pipeline import IntrusionDetectionPipeline
from pipelines.danger_zone_pipeline import DangerZonePipeline
from pipelines.face_recognition_pipeline import FaceRecognitionPipeline

class PipelineRegistry:
    def __init__(self):
        self._registry: Dict[str, Type[BaseVideoPipeline]] = {}
        self._instances: Dict[str, BaseVideoPipeline] = {}

    def register(self, name: str, pipeline_class: Type[BaseVideoPipeline]):
        self._registry[name] = pipeline_class

    def get_pipeline(self, name: str) -> BaseVideoPipeline:
        if name in self._instances:
            return self._instances[name]

        pipeline_class = self._registry.get(name)
        if not pipeline_class:
            raise ValueError(f"Pipeline '{name}' not found in registry.")
        
        # Instantiate and initialize the pipeline
        pipeline = pipeline_class()
        pipeline.initialize()
        self._instances[name] = pipeline
        return pipeline

    def get_available_pipelines(self) -> list:
        return list(self._registry.keys())

# Singleton instance
registry = PipelineRegistry()

# Register out-of-the-box pipelines
registry.register("intrusion_detection", IntrusionDetectionPipeline)
registry.register("danger_zone", DangerZonePipeline)

# ── Face Recognition pipeline (Ayush module) ─────────────────────────────────
registry.register("face_recognition", FaceRecognitionPipeline)
